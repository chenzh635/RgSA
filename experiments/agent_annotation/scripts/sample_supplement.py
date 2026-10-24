"""Expand the production annotation set to ~240 emails with the spec's
failure-mode strata (groundtruth_annotation_spec_2026-09-12.md §2).

Strata (priority order, one tag per email):
  mismatch = per-email Spearman(attn, rule) < 0        target ~60  (主战场)
  short    = exactly 2 sentences                        target ~40
             (1-sentence emails excluded: degenerate P@1=1 for every model,
              same reason as the Spearman exclusion rule; the original 150
              also excluded them)
  diffuse  = attention entropy > 1.5 and >= 6 sentences target ~40
  typical  = everything else                            fill to TOTAL

The existing 150 emails (annotation_data.json) are kept as-is and count
toward the targets; only the shortfall is sampled fresh from the pool
(tasks/_email_pool_metrics.json, produced by extract_email_pool.py from the
lambda_rule=3.0 seed-123 checkpoint — the paper's main model).

Writes:
  tasks/production_chunk_7..N.json  — blinded supplement emails (idx/text only)
  tasks/_key_stratum.json           — UPGRADED schema, now for every email:
      {batch, stratum (original quartile tag or None),
       failure_mode (mismatch/short/diffuse/typical), set (pilot30/prod150/supplement)}
  tasks/_sampling_report.json       — counts per stratum (for the paper)

Usage: python3 experiments/agent_annotation/scripts/sample_supplement.py
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
TASKS = os.path.join(HERE, '..', 'tasks')
HA = os.path.join(ROOT, 'experiments', 'human_annotation')

TARGETS = {'mismatch': 60, 'short': 40, 'diffuse': 40}
TOTAL = 240
SEED = 0
CHUNK_SIZE = 25
FIRST_NEW_CHUNK = 7


def failure_mode(m):
    sp = m.get('spearman')
    if sp is not None and sp < 0:
        return 'mismatch'
    if m['n_sentences'] == 2:
        return 'short'
    if m['entropy'] > 1.5 and m['n_sentences'] >= 6:
        return 'diffuse'
    return 'typical'


def main():
    with open(os.path.join(TASKS, '_email_pool_metrics.json')) as f:
        pool = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(HA, 'annotation_data.json')) as f:
        prod150 = json.load(f)['emails']
    with open(os.path.join(HA, 'pilot_30.json')) as f:
        pilot30 = json.load(f)['emails']

    existing = {e['email_idx']: e for e in prod150}
    pilot_ids = {e['email_idx'] for e in pilot30}
    missing = [i for i in existing if i not in pool]
    if missing:
        raise SystemExit(f'ERROR: {len(missing)} existing emails not found in pool '
                         f'(idx convention mismatch?): e.g. {missing[:5]}')

    # tag everything already selected
    key = {}
    counts = {t: 0 for t in TARGETS}
    counts['typical'] = 0
    for eid, e in existing.items():
        fm = failure_mode(pool[eid])
        counts[fm] += 1
        key[str(eid)] = {'batch': 'production_150', 'stratum': e['stratum'],
                         'failure_mode': fm, 'set': 'prod150'}
    for eid in pilot_ids:
        entry = key.get(str(eid))
        if entry:
            entry['set'] = 'pilot30+prod150'
        else:  # pilot email outside the 150 (not expected, but safe)
            key[str(eid)] = {'batch': 'pilot_30', 'stratum': None,
                             'failure_mode': failure_mode(pool[eid]),
                             'set': 'pilot30'}

    print('existing 150 failure-mode coverage:', counts)

    # sample the shortfall per stratum
    rng = random.Random(SEED)
    supplement = []
    used = set(existing) | pilot_ids
    for tag, target in TARGETS.items():
        need = max(0, target - counts[tag])
        cands = [eid for eid, m in pool.items()
                 if eid not in used and failure_mode(m) == tag]
        if len(cands) < need:
            print(f'WARNING: only {len(cands)} candidates for {tag}, need {need}')
            need = len(cands)
        picked = rng.sample(sorted(cands), need)
        used.update(picked)
        counts[tag] += need
        for eid in picked:
            supplement.append((eid, tag))
        print(f'  {tag}: +{need} (pool had {len(cands)})')

    need_typ = max(0, TOTAL - len(existing) - len(supplement))
    cands = [eid for eid, m in pool.items()
             if eid not in used and failure_mode(m) == 'typical'
             and pool[eid]['n_sentences'] >= 2]
    picked = rng.sample(sorted(cands), min(need_typ, len(cands)))
    counts['typical'] += len(picked)
    for eid in picked:
        supplement.append((eid, 'typical'))
    print(f'  typical: +{len(picked)}')

    supplement.sort(key=lambda t: t[0])

    # write blinded chunks
    for ci in range(0, len(supplement), CHUNK_SIZE):
        chunk = supplement[ci:ci + CHUNK_SIZE]
        n = FIRST_NEW_CHUNK + ci // CHUNK_SIZE
        doc = {
            'metadata': {
                'batch': f'production_chunk_{n}',
                'n': len(chunk),
                'note': 'Blinded supplement emails (spec failure-mode strata). '
                        'Sentences: idx/text only.',
            },
            'emails': [{
                'email_idx': eid,
                'n_sentences': pool[eid]['n_sentences'],
                'sentences': [{'idx': i, 'text': t}
                              for i, t in enumerate(pool[eid]['sentences'])],
            } for eid, _ in chunk],
        }
        path = os.path.join(TASKS, f'production_chunk_{n}.json')
        with open(path, 'w') as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        for eid, tag in chunk:
            key[str(eid)] = {'batch': f'production_chunk_{n}', 'stratum': None,
                             'failure_mode': tag, 'set': 'supplement'}
        print(f'production_chunk_{n}.json: {len(chunk)} emails')

    # leak check on new chunks
    for name in sorted(os.listdir(TASKS)):
        if name.startswith('_') or not name.endswith('.json'):
            continue
        with open(os.path.join(TASKS, name)) as f:
            txt = f.read()
        for bad in ('stratum', 'pred_prob', 'spearman', 'entropy', 'rule_score',
                    '"attn"', 'failure_mode'):
            assert bad not in txt, f'LEAK {bad} in {name}'

    with open(os.path.join(TASKS, '_key_stratum.json'), 'w') as f:
        json.dump(key, f, indent=1)
    report = {
        'total_emails': len(key),
        'targets': {**TARGETS, 'typical': f'fill to {TOTAL}', 'TOTAL': TOTAL},
        'final_counts': counts,
        'n_existing150': len(existing),
        'n_supplement': len(supplement),
        'seed': SEED,
        'note': 'failure_mode from lambda_rule=3.0 seed-123 sparse checkpoint; '
                'original 150 stratified by archive-checkpoint spearman quartiles.',
    }
    with open(os.path.join(TASKS, '_sampling_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('\nfinal counts:', counts, '| total:', len(key))
    print('blinding sanity: OK | sidecar + report written')


if __name__ == '__main__':
    main()
