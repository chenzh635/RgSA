"""Merge 3 agent passes into majority-vote gold + disagreement queue (production).

Two modes:

1) Build majority gold and the adjudication queue:
   python3 scripts/merge_and_adjudicate.py \
       --pass-a outputs/agent_pass_a_production_chunk_*.json \
       --pass-b outputs/agent_pass_b_production_chunk_*.json \
       --pass-c outputs/agent_pass_c_production_chunk_*.json

   Writes:
     outputs/gold_majority.json          — export-format gold (run_gt_evaluation
                                           --gold compatible); only sentences
                                           with a >=2/3 majority; stratum =
                                           failure_mode tag re-attached from
                                           tasks/_key_stratum.json
     outputs/production_disagreements.csv — no-majority items with all three
                                           labels + reasons (input to the
                                           adjudicator agent, see
                                           prompts/adjudicator_prompt.md)
     outputs/groundtruth_oracle_labels.jsonl — per-sentence record with final
                                           label + each judge's raw label and
                                           confidence (spec §11.1)

2) After the adjudicator finishes, fold its verdicts in:
   python3 scripts/merge_and_adjudicate.py \
       --apply-adjudication outputs/adjudication.json

   Writes outputs/gold_final.json = gold_majority + adjudicated items.
"""

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'outputs'
KEY_FILE = HERE.parent / 'tasks' / '_key_stratum.json'
CATS = {'Y', 'N', '?'}


def load_pass(patterns):
    """Merge one or more export-format JSON files into {(email, sent): row}."""
    ann = {}
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)) if ('*' in pat or '?' in pat) else [pat])
    if not files:
        sys.exit(f'ERROR: no files match {patterns}')
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        for row in data['annotations']:
            label = str(row.get('label', '')).strip()
            if label not in CATS:
                sys.exit(f'ERROR: bad label {label!r} in {fp} '
                         f'(email {row.get("email_id")}, sent {row.get("sentence_idx")})')
            key = (int(row['email_id']), int(row['sentence_idx']))
            if key in ann and ann[key]['label'] != label:
                sys.exit(f'ERROR: duplicate conflicting entry {key} within pass')
            ann[key] = {'label': label,
                        'reason': row.get('reason', ''),
                        'text': row.get('sentence_text', ''),
                        'confidence': row.get('confidence', 'unspecified')}
    return ann, files


def write_md_report(gold, adj, n_added):
    """annotation_agreement_report.md — spec §11.3 deliverable (ensemble part)."""
    rows = gold['annotations']
    dist = Counter(r['label'] for r in rows)
    conf = Counter(r.get('confidence', 'unspecified') for r in rows)
    emails = {r['email_id'] for r in rows}
    covered = len({r['email_id'] for r in rows if r['label'] == 'Y'})
    by_stratum = Counter(r.get('stratum', '') for r in rows)
    n = len(rows)
    lines = [
        '# 智能体集成标注 · 一致性报告（ensemble part）',
        '',
        f'- 最终 gold：{n} 句 / {len(emails)} 封（`gold_final.json`）',
        f'- 标签分布：{dict(dist)}（Y 占 {dist.get("Y", 0) / max(n, 1):.3f}）',
        f'- confidence 分布：{dict(conf)}（low 占 {conf.get("low", 0) / max(n, 1):.3f}）',
        f'- oracle 覆盖率（≥1 Y 的邮件占比）：{covered}/{len(emails)}'
        f' = {covered / max(len(emails), 1):.3f}',
        f'- 仲裁折入：{n_added} 句（无多数票项）',
        f'- 句子数按分层：{dict(by_stratum)}',
        '',
        '裁判间 κ、agent-vs-human κ、位置偏差结论见 '
        '`calibration_report.json` 与 `position_bias_report.json`',
        '（pilot 阶段产出；数值应同步摘录到本文件下方）。',
        '',
        '## 待协调者手填（跑完 pilot 校准后）',
        '',
        '- 裁判间 Fleiss κ（3 agent）：＿＿＿',
        '- agent-vs-human 平均 κ（G1）：＿＿＿',
        '- κ(chan,li) 基线：0.398（excl-4860：0.471）',
        '- 位置偏差（顺序 vs 打乱翻转率）：＿＿＿',
        '- 裁判模型/厂商：＿＿＿（异质性披露）',
    ]
    with open(OUT / 'annotation_agreement_report.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pass-a', nargs='+')
    ap.add_argument('--pass-b', nargs='+')
    ap.add_argument('--pass-c', nargs='+')
    ap.add_argument('--apply-adjudication')
    args = ap.parse_args()

    with open(KEY_FILE) as f:
        key = json.load(f)

    def stratum_of(email_id):
        entry = key.get(str(email_id))
        if not entry:
            return ''
        return entry.get('failure_mode') or entry.get('stratum') or ''

    gold_majority_path = OUT / 'gold_majority.json'

    # ---------------- mode 2: apply adjudication ----------------
    if args.apply_adjudication:
        with open(gold_majority_path) as f:
            gold = json.load(f)
        have = {(a['email_id'], a['sentence_idx']) for a in gold['annotations']}
        adj, files = load_pass([args.apply_adjudication])
        added = 0
        for (eid, sidx), row in sorted(adj.items()):
            if (eid, sidx) in have:
                print(f'WARNING: adjudication overlaps majority item {eid}/{sidx}; skipped')
                continue
            gold['annotations'].append({
                'email_id': eid, 'sentence_idx': sidx,
                'sentence_text': row['text'], 'stratum': stratum_of(eid),
                'label': row['label'], 'reason': f"[adjudicated] {row['reason']}",
            })
            added += 1
        gold['annotator'] = 'agent_ensemble_final'
        gold['n_emails'] = len({a['email_id'] for a in gold['annotations']})
        gold['annotations'].sort(key=lambda a: (a['email_id'], a['sentence_idx']))
        final_path = OUT / 'gold_final.json'
        with open(final_path, 'w') as f:
            json.dump(gold, f, ensure_ascii=False, indent=1)

        # patch the jsonl with adjudicated final labels
        jsonl_path = OUT / 'groundtruth_oracle_labels.jsonl'
        if jsonl_path.exists():
            rows = []
            with open(jsonl_path) as f:
                for line in f:
                    rows.append(json.loads(line))
            for r in rows:
                k = (r['email_id'], r['sentence_idx'])
                if r['final_label'] is None and k in adj:
                    r['final_label'] = adj[k]['label']
                    r['source'] = 'adjudicated'
            with open(jsonl_path, 'w') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

        write_md_report(gold, adj, added)
        print(f'adjudication merged: +{added} items -> {final_path} '
              f'({len(gold["annotations"])} total)')
        return

    # ---------------- mode 1: majority + disagreements ----------------
    if not (args.pass_a and args.pass_b and args.pass_c):
        ap.error('need --pass-a/--pass-b/--pass-c (or use --apply-adjudication)')

    passes = []
    for name, pats in (('pass_a', args.pass_a), ('pass_b', args.pass_b),
                       ('pass_c', args.pass_c)):
        ann, files = load_pass(pats)
        passes.append((name, ann))
        print(f'{name}: {len(ann)} sentences from {len(files)} file(s)')

    keysets = [set(a) for _, a in passes]
    if not (keysets[0] == keysets[1] == keysets[2]):
        union = set().union(*keysets)
        for (name, ann), ks in zip(passes, keysets):
            miss = sorted(union - ks)[:10]
            if miss:
                print(f'WARNING {name} missing {len(union - ks)} items, e.g. {miss}')
        sys.exit('ERROR: passes disagree on item coverage; fix before merging')

    common = sorted(keysets[0])
    gold_rows, disag_rows, jsonl_rows = [], [], []
    for k in common:
        eid, sidx = k
        labs = [passes[i][1][k]['label'] for i in range(3)]
        confs = [passes[i][1][k]['confidence'] for i in range(3)]
        cnt = Counter(labs)
        top = cnt.most_common(2)
        text = next((passes[i][1][k]['text'] for i in range(3)
                     if passes[i][1][k]['text']), '')
        votes = {passes[i][0]: {'label': labs[i], 'confidence': confs[i],
                                'reason': passes[i][1][k]['reason']}
                 for i in range(3)}
        base = {'email_id': eid, 'sentence_idx': sidx, 'sentence_text': text,
                'stratum': stratum_of(eid), 'votes': votes}
        if len(top) == 1 or top[0][1] > top[1][1]:
            maj_rows = [i for i in range(3) if labs[i] == top[0][0]]
            reason = next((passes[i][1][k]['reason'] for i in maj_rows
                           if passes[i][1][k]['reason']), '')
            conf = passes[maj_rows[0]][1][k]['confidence']
            gold_rows.append({
                'email_id': eid, 'sentence_idx': sidx, 'sentence_text': text,
                'stratum': stratum_of(eid), 'label': top[0][0],
                'confidence': conf, 'reason': reason,
            })
            jsonl_rows.append({**base, 'final_label': top[0][0],
                               'source': 'majority'})
        else:
            jsonl_rows.append({**base, 'final_label': None,
                               'source': 'pending_adjudication'})
            disag_rows.append({
                'email_id': eid, 'sentence_idx': sidx, 'sentence_text': text,
                **{passes[i][0]: labs[i] for i in range(3)},
                **{f'reason_{passes[i][0]}': passes[i][1][k]['reason']
                   for i in range(3)},
            })

    gold = {
        'annotator': 'agent_ensemble_majority',
        'batch': 'production_150',
        'n_emails': len({r['email_id'] for r in gold_rows}),
        'annotations': gold_rows,
    }
    OUT.mkdir(exist_ok=True)
    with open(gold_majority_path, 'w') as f:
        json.dump(gold, f, ensure_ascii=False, indent=1)

    disag_path = OUT / 'production_disagreements.csv'
    with open(disag_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(disag_rows[0].keys()) if disag_rows else
                           ['email_id', 'sentence_idx', 'sentence_text', 'pass_a',
                            'pass_b', 'pass_c', 'reason_pass_a', 'reason_pass_b',
                            'reason_pass_c'])
        w.writeheader()
        w.writerows(disag_rows)

    jsonl_path = OUT / 'groundtruth_oracle_labels.jsonl'
    with open(jsonl_path, 'w') as f:
        for r in jsonl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    dist = Counter(r['label'] for r in gold_rows)
    conf_dist = Counter(r.get('confidence', 'unspecified') for r in gold_rows)
    emails = {r['email_id'] for r in gold_rows}
    covered = len({r['email_id'] for r in gold_rows if r['label'] == 'Y'})
    print(f'\nmajority gold: {len(gold_rows)}/{len(common)} sentences '
          f'({dict(dist)}) -> {gold_majority_path}')
    print(f'confidence dist: {dict(conf_dist)} '
          f'(low share {conf_dist.get("low", 0) / max(len(gold_rows), 1):.3f})')
    print(f'oracle coverage: {covered}/{len(emails)} emails have >=1 Y '
          f'({covered / max(len(emails), 1):.3f})')
    print(f'per-judge raw labels -> {jsonl_path}')
    print(f'no-majority items: {len(disag_rows)} -> {disag_path}')
    print('next: run adjudicator agent with prompts/adjudicator_prompt.md, then\n'
          '  python3 scripts/merge_and_adjudicate.py --apply-adjudication outputs/adjudication.json')


if __name__ == '__main__':
    main()
