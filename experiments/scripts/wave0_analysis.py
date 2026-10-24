"""Wave 0 analyses: ground-truth-free evidence on localization quality.

Four analyses, all inference-only (target: AsiaCCS 2027 revision):

  1. stability  - selection stability across seeds: top-1 agreement and
                  top-3 Jaccard between the 3 sparse seeds (ABMIL seeds as
                  contrast) over the full test set
  2. loo        - leave-one-sentence-out importance vs attention rank on the
                  150 annotation-target emails (pooling decomposition, same
                  protocol as src/evaluate_faithfulness.py): per-sentence
                  delta-logit = logit_full - logit(sentence weight zeroed,
                  rest renormalized); Spearman(delta, attn) and top-1 overlap
  3. benign     - attention behaviour on benign emails (entropy, top-1 rule
                  score/keyword hit, misclassified subset) vs phishing emails
                  -> motivates conditional localization
  4. cognitive  - Cognitive-TP-style keyword-hit rate at top-k for
                  sparse/abmil/ai2tale. NOTE: partially circular for ours
                  (trained toward rule scores); non-circular for baselines.

Usage:
    python experiments/scripts/wave0_analysis.py [--device auto]
        [--parts stability,loo,benign,cognitive] [--max-batches N]
        [--output experiments/wave0_analysis.json]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_faithfulness import SEEDS, build_model, ckpt, load_config, load_state
from src.dataset import create_dataloaders
from src.evaluate_faithfulness import make_classifier_fn
from src.rules import batch_score, load_keywords

EPS = 1e-8
KCOLLECT = 5
ANNOTATION_DATA = os.path.join(project_root, 'experiments', 'human_annotation',
                               'annotation_data.json')


# ------------------------------------------------------------ helpers
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def n_valid_of(sl_row, w_row):
    return min(int((sl_row > 0).sum()), w_row.shape[0])


def norm_entropy(w):
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def forward_all(model, kind, batch, device):
    """Return (logits, attn_weights, h) as numpy for one batch."""
    sent_lengths = batch['sent_lengths'].to(device)
    if kind == 'cnn_sparse':
        out, attn_w, _, h = model(batch['tokens'].to(device), sent_lengths,
                                  temperature=0.1)
    elif kind == 'cnn_abmil':
        out, attn_w, h = model(batch['tokens'].to(device), sent_lengths)
    elif kind == 'ai2tale':
        out, attn_w, h = model(batch['tokens'].to(device), sent_lengths)
    else:
        raise ValueError(kind)
    return (out.cpu().numpy().astype(np.float64),
            attn_w.detach().cpu().numpy().astype(np.float64),
            h.detach().cpu().numpy().astype(np.float32))


def load_model_entry(kind, path, device):
    model = build_model(kind)
    model.load_state_dict(load_state(path), strict=False)
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------ collection
def collect(model, kind, loader, device, keywords, max_batches=None):
    """One pass over the test set; per-email stats keyed by email idx."""
    stats = {}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            logits, w_all, _ = forward_all(model, kind, batch, device)
            rule_scores = batch_score(batch['raw_sentences'], keywords)
            sl = batch['sent_lengths'].numpy()
            for b, eid in enumerate(batch['idx']):
                eid = int(eid)
                nv = n_valid_of(sl[b], w_all[b])
                if nv == 0:
                    continue
                w = w_all[b, :nv].copy()
                s = w.sum()
                if s <= EPS:
                    continue
                w /= s
                order = np.argsort(-w, kind='stable')
                top = [int(i) for i in order[:min(KCOLLECT, nv)]]
                rs = rule_scores[b] if rule_scores.shape[1] >= nv \
                    else np.pad(rule_scores[b], (0, nv - rule_scores.shape[1]))
                stats[eid] = {
                    'label': int(batch['label'][b]),
                    'prob': float(sigmoid(logits[b])),
                    'logit': float(logits[b]),
                    'n_valid': nv,
                    'entropy': norm_entropy(w),
                    'top1_weight': float(w[order[0]]),
                    'top': top,
                    'rule_top': [float(rs[i]) for i in top],
                    'n_rule_pos': int((rs[:nv] > 0).sum()),
                }
    return stats


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def stability_from(collections):
    """collections: {tag: stats}. Pairwise top-3 Jaccard + top-1 agreement."""
    tags = list(collections)
    eids = sorted(set.intersection(*[set(c) for c in collections.values()]))
    out = {'n_emails': len(eids), 'tags': tags,
           'pairwise_jaccard_top3': {}, 'top1_all_agree': None,
           'by_label': {}}
    pairs = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            vals = [jaccard(collections[tags[i]][e]['top'][:3],
                            collections[tags[j]][e]['top'][:3]) for e in eids]
            out['pairwise_jaccard_top3'][f'{tags[i]}~{tags[j]}'] = {
                'mean': float(np.mean(vals)),
                'p10': float(np.percentile(vals, 10)),
                'frac_ge_0.99': float(np.mean(np.array(vals) >= 0.99 - 1e-9)),
            }
            t1 = [collections[tags[i]][e]['top'][0]
                  == collections[tags[j]][e]['top'][0] for e in eids]
            pairs.append((tags[i], tags[j], float(np.mean(t1))))
    out['pairwise_top1_agreement'] = [
        {'pair': f'{a}~{b}', 'mean': v} for a, b, v in pairs]
    all_agree = [len({collections[t][e]['top'][0] for t in tags}) == 1
                 for e in eids]
    out['top1_all_agree'] = float(np.mean(all_agree))
    for lab, name in [(1, 'phishing'), (0, 'benign')]:
        sub = [e for e in eids if collections[tags[0]][e]['label'] == lab]
        if sub:
            out['by_label'][name] = {
                'n': len(sub),
                'top1_all_agree': float(np.mean(
                    [len({collections[t][e]['top'][0] for t in tags}) == 1
                     for e in sub])),
                'jaccard_top3_mean': float(np.mean(
                    [jaccard(collections[tags[i]][e]['top'][:3],
                             collections[tags[j]][e]['top'][:3])
                     for e in sub for i in range(len(tags))
                     for j in range(i + 1, len(tags))])),
            }
    return out


def benign_report(stats):
    """Attention behaviour: benign vs phishing, split benign by correctness."""
    rows = list(stats.values())
    ph = [r for r in rows if r['label'] == 1]
    be = [r for r in rows if r['label'] == 0]
    be_fp = [r for r in be if r['prob'] >= 0.5]
    be_tn = [r for r in be if r['prob'] < 0.5]

    def block(rs):
        if not rs:
            return {'n': 0}
        return {
            'n': len(rs),
            'mean_prob': float(np.mean([r['prob'] for r in rs])),
            'mean_entropy': float(np.mean([r['entropy'] for r in rs])),
            'mean_top1_weight': float(np.mean([r['top1_weight'] for r in rs])),
            'mean_top1_rule': float(np.mean([r['rule_top'][0] for r in rs])),
            'top1_keyword_hit': float(np.mean(
                [r['rule_top'][0] > 0 for r in rs])),
            'mean_n_valid': float(np.mean([r['n_valid'] for r in rs])),
        }

    return {'phishing': block(ph), 'benign_all': block(be),
            'benign_misclassified': block(be_fp),
            'benign_correct': block(be_tn)}


def cognitive_report(stats):
    """Cognitive-TP style: top-k selections hitting keyword-bearing sentences.

    Scope: phishing emails. Partially circular for rule-trained models.
    """
    ph = [r for r in stats.values() if r['label'] == 1]
    out = {'n_phishing': len(ph)}
    if not ph:
        return out
    for k in (1, 3, 5):
        hit, frac = [], []
        for r in ph:
            kk = min(k, len(r['top']))
            hits = sum(1 for x in r['rule_top'][:kk] if x > 0)
            hit.append(1 if hits > 0 else 0)
            frac.append(hits / kk)
        out[f'hit@{k}'] = float(np.mean(hit))
        out[f'hit_frac@{k}'] = float(np.mean(frac))
    # context: how many phishing emails have ANY keyword sentence
    out['emails_with_any_keyword'] = float(np.mean(
        [r['n_rule_pos'] > 0 for r in ph]))
    return out


# ------------------------------------------------------------ LOO
def loo_analysis(model, kind, loader, device, keywords, wanted_ids=None,
                 max_batches=None, classify_attr='classifier',
                 require_phishing=False):
    """Per-sentence deletion importance vs attention.

    wanted_ids: restrict to these emails; None = all emails (optionally
    label==1 only via require_phishing).
    """
    classify_fn = make_classifier_fn(getattr(model, classify_attr), device)
    remaining = set(wanted_ids) if wanted_ids is not None else None
    per_email = {}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            if remaining is not None and not remaining:
                break
            logits, w_all, h_all = forward_all(model, kind, batch, device)
            rule_scores = batch_score(batch['raw_sentences'], keywords)
            sl = batch['sent_lengths'].numpy()
            for b, eid in enumerate(batch['idx']):
                eid = int(eid)
                if remaining is not None:
                    if eid not in remaining:
                        continue
                    remaining.discard(eid)
                elif require_phishing and int(batch['label'][b]) != 1:
                    continue
                nv = n_valid_of(sl[b], w_all[b])
                if nv < 2:
                    per_email[eid] = {'excluded': 'too_short', 'n_valid': nv}
                    continue
                w = w_all[b, :nv].copy()
                s = w.sum()
                if s <= EPS:
                    per_email[eid] = {'excluded': 'no_selection', 'n_valid': nv}
                    continue
                w /= s
                full = float(logits[b])
                h_b = h_all[b, :nv]
                deltas = np.full(nv, np.nan)
                for j in range(nv):
                    rest = 1.0 - w[j]
                    if rest <= EPS:
                        continue
                    w_new = w.copy()
                    w_new[j] = 0.0
                    w_new /= rest
                    z = (h_b * w_new[:, None]).sum(axis=0)
                    deltas[j] = full - classify_fn(z)
                defined = ~np.isnan(deltas)
                entry = {
                    'n_valid': nv,
                    'logit_full': full,
                    'prob_full': float(sigmoid(full)),
                    'n_defined': int(defined.sum()),
                    'delta_std': float(np.nanstd(deltas))
                    if defined.any() else 0.0,
                    'top1_attn_concentrated': bool(not defined[w.argmax()]),
                    'top1_attn': int(w.argmax()),
                }
                rs = rule_scores[b, :nv] if rule_scores.shape[1] >= nv else \
                    np.pad(rule_scores[b], (0, nv - rule_scores.shape[1]))
                entry['spearman_rule_delta'] = _spearman(rs[defined],
                                                         deltas[defined])
                entry['spearman_attn_delta'] = _spearman(w[defined],
                                                         deltas[defined])
                if defined.any():
                    entry['top1_loo'] = int(np.nanargmax(deltas))
                    entry['top1_match'] = bool(
                        entry['top1_loo'] == entry['top1_attn'])
                    entry['top1_delta_positive'] = bool(
                        deltas[entry['top1_attn']] > 0) \
                        if defined[entry['top1_attn']] else None
                per_email[eid] = entry
    return per_email


def _spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_loo(per_email):
    ents = [e for e in per_email.values() if 'excluded' not in e]

    def mean_of(key, subset=None):
        rows = subset if subset is not None else ents
        vals = [e[key] for e in rows if e.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    # tie-robust subset: enough sentences and non-degenerate deltas
    filtered = [e for e in ents
                if e.get('n_defined', 0) >= 5 and e.get('delta_std', 0) > 0.1]
    out = {
        'n_emails': len(per_email),
        'n_excluded': len(per_email) - len(ents),
        'n_filtered': len(filtered),
        'n_top1_concentrated': sum(
            1 for e in ents if e['top1_attn_concentrated']),
        'top1_match_rate': mean_of('top1_match'),
        'top1_match_rate_filtered': mean_of('top1_match', filtered),
        'mean_spearman_attn_delta': mean_of('spearman_attn_delta'),
        'mean_spearman_attn_delta_filtered': mean_of('spearman_attn_delta',
                                                     filtered),
        'mean_spearman_rule_delta': mean_of('spearman_rule_delta'),
        'mean_spearman_rule_delta_filtered': mean_of('spearman_rule_delta',
                                                     filtered),
        'frac_top1_delta_positive': mean_of('top1_delta_positive'),
    }
    return out


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--parts', default='stability,loo,benign,cognitive')
    ap.add_argument('--max-batches', type=int, default=None)
    ap.add_argument('--loo-ids', type=int, nargs='*', default=None,
                    help='debug override: email ids for the LOO part')
    ap.add_argument('--loo-scope', default='target',
                    choices=['target', 'phishing'],
                    help='target = 150 annotation emails; '
                         'phishing = all test phishing emails')
    ap.add_argument('--output', default=os.path.join(
        project_root, 'experiments', 'wave0_analysis.json'))
    args = ap.parse_args()
    parts = set(args.parts.split(','))

    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}', flush=True)

    config = load_config()
    _, test_loader = create_dataloaders(config)
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)

    if args.loo_ids:
        wanted_ids = list(args.loo_ids)
    else:
        with open(ANNOTATION_DATA) as f:
            wanted_ids = [e['email_idx'] for e in json.load(f)['emails']]
    print(f'test emails: {len(test_loader.dataset)}, '
          f'annotation-target emails: {len(wanted_ids)}', flush=True)

    entries = {
        'sparse': [('Sparse seed{}'.format(s),
                    ckpt('multi_seed_sparse_seed{}/sparse_best_model.pt'
                         .format(s))) for s in SEEDS],
        'abmil': [('ABMIL seed{}'.format(s),
                   ckpt('multi_seed_abmil_seed{}/abmil_best_model.pt'
                        .format(s))) for s in SEEDS],
        'ai2tale': [('AI2TALE',
                     ckpt('archive/ai2tale/ai2tale_best_model.pt'))],
    }
    kinds = {'sparse': 'cnn_sparse', 'abmil': 'cnn_abmil',
             'ai2tale': 'ai2tale'}

    results = {'meta': {'device': str(device),
                        'parts': sorted(parts),
                        'note_cognitive': ('keyword-hit metric is partially '
                                           'circular for the rule-trained '
                                           'sparse model; non-circular for '
                                           'baselines')}}

    # ---- passes needed by multiple parts -------------------------------
    need_collect = parts & {'stability', 'benign', 'cognitive'}
    collections = {}
    if need_collect:
        groups = []
        if 'stability' in parts:
            groups += entries['sparse'] + entries['abmil']
        else:
            groups += [('Sparse seed123', entries['sparse'][0][1]),
                       ('ABMIL seed123', entries['abmil'][0][1])]
        if 'benign' in parts or 'cognitive' in parts:
            groups += entries['ai2tale']
        for tag, path in groups:
            print(f'[collect] {tag} ...', flush=True)
            model = load_model_entry(kinds['sparse'] if 'Sparse' in tag
                                     else ('cnn_abmil' if 'ABMIL' in tag
                                           else 'ai2tale'),
                                     path, device)
            collections[tag] = collect(model, kinds['sparse'] if 'Sparse' in tag
                                       else ('cnn_abmil' if 'ABMIL' in tag
                                             else 'ai2tale'),
                                       test_loader, device, keywords,
                                       args.max_batches)
            del model
            print(f'    {len(collections[tag])} emails collected', flush=True)

    if 'stability' in parts:
        print('[stability] sparse x3 seeds ...', flush=True)
        sparse_cols = {t: collections[t] for t in collections
                       if t.startswith('Sparse')}
        results['stability_sparse'] = stability_from(sparse_cols)
        abmil_cols = {t: collections[t] for t in collections
                      if t.startswith('ABMIL')}
        if len(abmil_cols) == len(SEEDS):
            results['stability_abmil_contrast'] = stability_from(abmil_cols)

    if 'benign' in parts:
        print('[benign] attention behaviour ...', flush=True)
        results['benign'] = {}
        for tag in collections:
            results['benign'][tag] = benign_report(collections[tag])

    if 'cognitive' in parts:
        print('[cognitive] keyword-hit at top-k ...', flush=True)
        results['cognitive'] = {}
        for tag in collections:
            results['cognitive'][tag] = cognitive_report(collections[tag])

    if 'loo' in parts:
        loo_targets = [
            ('Sparse seed123', 'cnn_sparse', entries['sparse'][0][1],
             'classifier'),
            ('ABMIL seed123', 'cnn_abmil', entries['abmil'][0][1],
             'classifier'),
            ('AI2TALE', 'ai2tale', entries['ai2tale'][0][1], 'predictor'),
        ]
        if args.loo_scope == 'target':
            wanted, require_phishing, out_key = wanted_ids, False, 'loo'
        else:
            wanted, require_phishing, out_key = None, True, 'loo_population'
        results[out_key] = {}
        for tag, kind, path, attr in loo_targets:
            print(f'[loo:{args.loo_scope}] {tag} ...', flush=True)
            model = load_model_entry(kind, path, device)
            per_email = loo_analysis(model, kind, test_loader, device,
                                     keywords, wanted, args.max_batches,
                                     classify_attr=attr,
                                     require_phishing=require_phishing)
            del model
            correct_pe = {k: v for k, v in per_email.items()
                          if 'excluded' not in v and v['prob_full'] >= 0.5}
            results[out_key][tag] = {
                'summary': summarize_loo(per_email),
                'summary_correct': summarize_loo(correct_pe),
                'per_email': {str(k): v for k, v in per_email.items()},
            }

    # merge with previous results so subset re-runs keep other parts
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                prev = json.load(f)
            merged = dict(prev)
            merged.update(results)
            results = merged
        except (json.JSONDecodeError, OSError):
            pass
    with open(args.output, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\nsaved -> {args.output}')

    # ---- console summary -----------------------------------------------
    if 'stability' in parts and 'stability_sparse' in results:
        st = results['stability_sparse']
        print(f"\nstability(sparse): top1 all-agree {st['top1_all_agree']:.3f} "
              f"over {st['n_emails']} emails")
        for pair, v in st['pairwise_jaccard_top3'].items():
            print(f"  J@3 {pair}: {v['mean']:.3f} "
                  f"(p10 {v['p10']:.3f}, >=0.99: {v['frac_ge_0.99']:.3f})")
        if 'stability_abmil_contrast' in results:
            sa = results['stability_abmil_contrast']
            print(f"stability(abmil contrast): top1 all-agree "
                  f"{sa['top1_all_agree']:.3f}")
    if 'loo' in parts:
        loo_key = 'loo_population' if args.loo_scope == 'phishing' else 'loo'
        for tag, lo_entry in results[loo_key].items():
            for skey, label in [('summary', 'all'),
                                ('summary_correct', 'correct-only')]:
                lo = lo_entry[skey]
                print(f"\nloo({tag}, {label}): {lo['n_emails']} emails "
                      f"({lo['n_excluded']} excluded; {lo['n_filtered']} pass "
                      f"tie filter), top1 concentrated in "
                      f"{lo['n_top1_concentrated']}")
                for key in ('top1_match_rate', 'top1_match_rate_filtered',
                            'mean_spearman_attn_delta',
                            'mean_spearman_attn_delta_filtered',
                            'mean_spearman_rule_delta',
                            'frac_top1_delta_positive'):
                    v = lo.get(key)
                    print(f"  {key}: {v if v is None else round(v, 3)}")
    if 'benign' in parts:
        for tag, rep in results['benign'].items():
            b = rep['benign_all']
            p = rep['phishing']
            if b.get('n', 0) == 0:
                continue
            msg = (f"\nbenign({tag}): H={b['mean_entropy']:.3f} "
                   f"top1w={b['mean_top1_weight']:.3f} "
                   f"top1hit={b['top1_keyword_hit']:.3f}; ")
            if p.get('n', 0):
                msg += (f"(phishing: H={p['mean_entropy']:.3f} "
                        f"top1w={p['mean_top1_weight']:.3f} "
                        f"top1hit={p['top1_keyword_hit']:.3f}); ")
            else:
                msg += "(phishing: none in this run); "
            msg += f"misclassified benign n={rep['benign_misclassified']['n']}"
            print(msg)
    if 'cognitive' in parts:
        for tag, rep in results['cognitive'].items():
            if rep.get('n_phishing', 0) == 0:
                print(f"cognitive({tag}): no phishing emails in this run")
                continue
            print(f"cognitive({tag}): hit@1 {rep['hit@1']:.3f}, "
                  f"hit@3 {rep['hit@3']:.3f}, hit@5 {rep['hit@5']:.3f}")


if __name__ == '__main__':
    main()
