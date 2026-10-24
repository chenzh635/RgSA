"""Ground-truth sentence-localization evaluation for all 7 models.

Extracts each model's per-sentence selection scores on the annotated emails,
then scores localization quality against human gold annotations
(Precision@k / Recall@k / F1@k / IoU@k / usability), replacing the circular
Spearman(attn, rule_scores) metric.

Two stages (scores cached so statistics can be re-run without inference):
  1. extract -> experiments/gt_scores.json   (needs model checkpoints)
  2. score   -> experiments/gt_localization_results.json + printed table

Usage:
    python experiments/scripts/run_gt_evaluation.py \
        --gold experiments/human_annotation/annotations_*.csv [...] \
        [--models sparse,abmil,bert,sparse_bert,ai2tale,snorkel,rules,random] \
        [--extract-only] [--scores-only] [--device auto] \
        [--by-stratum]

Sentence-index convention: position i in the model's score vector
corresponds to raw_sentences[i] — identical to prepare_human_annotation.py,
so gold sentence indices align.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_faithfulness import SEEDS, build_model, ckpt, load_config, load_state
from src.dataset import create_dataloaders
from src.evaluate_sentence_localization import (
    KS_DEFAULT, evaluate_scores, format_table, load_gold,
    bootstrap_ci, paired_diff_ci,
)


def _forward_weights(model, kind, batch, device):
    """Return selection weights (B, S) on raw-sentence positions."""
    sent_lengths = batch['sent_lengths'].to(device)
    if kind == 'cnn_sparse':
        _, attn_w, _, _ = model(batch['tokens'].to(device), sent_lengths,
                                temperature=0.1)
    elif kind == 'cnn_abmil':
        _, attn_w, _ = model(batch['tokens'].to(device), sent_lengths)
    elif kind == 'bert':
        _, attn_w, _ = model(batch['raw_sentences'], sent_lengths)
    elif kind == 'bert_sparse':
        _, attn_w, _, _ = model(batch['raw_sentences'], sent_lengths,
                                temperature=0.1)
    elif kind == 'ai2tale':
        _, attn_w, _ = model(batch['tokens'].to(device), sent_lengths)
    else:
        raise ValueError(kind)
    return attn_w.detach().cpu().numpy().astype(np.float64)


def extract_scores_model(model, kind, test_loader, device, wanted_ids):
    scores = {}
    remaining = set(wanted_ids)
    with torch.no_grad():
        for batch in test_loader:
            raw_sentences = batch['raw_sentences']
            w_all = _forward_weights(model, kind, batch, device)
            for b, eid in enumerate(batch['idx']):
                if eid not in remaining:
                    continue
                n_raw = len(raw_sentences[b])
                scores[eid] = w_all[b, :n_raw].tolist()
                remaining.discard(eid)
            if not remaining:
                break
    return scores


def extract_scores_snorkel(model, test_loader, device, wanted_ids):
    scores = {}
    remaining = set(wanted_ids)
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            for b, eid in enumerate(batch['idx']):
                if eid not in remaining:
                    continue
                n_raw = len(batch['raw_sentences'][b])
                logits = model(tokens[b, :n_raw, :]).cpu().numpy().astype(np.float64)
                stable = logits - logits.max()
                w = np.exp(stable)
                scores[eid] = (w / w.sum()).tolist()
                remaining.discard(eid)
            if not remaining:
                break
    return scores


def extract_scores_random(test_loader, wanted_ids, seed=0):
    """Random-selection baseline (spec §8 comparison matrix); deterministic."""
    rng = np.random.default_rng(seed)
    scores = {}
    remaining = set(wanted_ids)
    for batch in test_loader:
        for b, eid in enumerate(batch['idx']):
            if eid not in remaining:
                continue
            n_raw = len(batch['raw_sentences'][b])
            scores[eid] = rng.random(n_raw).tolist()
            remaining.discard(eid)
        if not remaining:
            break
    return scores


def extract_scores_rules(test_loader, config, wanted_ids):
    from src.rules import batch_score, load_keywords
    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)
    scores = {}
    remaining = set(wanted_ids)
    for batch in test_loader:
        rule_scores = batch_score(batch['raw_sentences'], keywords)
        for b, eid in enumerate(batch['idx']):
            if eid not in remaining:
                continue
            n_raw = len(batch['raw_sentences'][b])
            scores[eid] = rule_scores[b, :n_raw].tolist()
            remaining.discard(eid)
        if not remaining:
            break
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold', nargs='+', required=True,
                    help='annotation CSVs (1 = single rater, 2-3 = majority)')
    ap.add_argument('--models', default='sparse,abmil,bert,sparse_bert,'
                                         'ai2tale,snorkel,rules,random')
    ap.add_argument('--scores-file', default=os.path.join(
        project_root, 'experiments', 'gt_scores.json'))
    ap.add_argument('--output', default=os.path.join(
        project_root, 'experiments', 'gt_localization_results.json'))
    ap.add_argument('--extract-only', action='store_true')
    ap.add_argument('--scores-only', action='store_true',
                    help='skip extraction, use cached scores file')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--ks', default='1,3,5')
    ap.add_argument('--mode', default='strict', choices=['strict', 'exclude_q'])
    ap.add_argument('--reference', default='Sparse ABMIL (Ours)',
                    help='model for paired-difference CIs')
    ap.add_argument('--by-stratum', action='store_true')
    args = ap.parse_args()

    ks = tuple(int(x) for x in args.ks.split(','))
    gold = load_gold(args.gold)
    wanted_ids = sorted({eid for (eid, _) in gold['labels']})
    print(f"gold: {gold['n_items']} sentences over {len(wanted_ids)} emails "
          f"({gold['n_raters']} rater(s), {gold['n_ties']} ties)")

    registry = {
        'sparse': ('Sparse ABMIL (Ours)', 'cnn_sparse',
                   [ckpt(f'multi_seed_sparse_seed{s}/sparse_best_model.pt') for s in SEEDS]),
        'abmil': ('ABMIL (CNN)', 'cnn_abmil',
                  [ckpt(f'multi_seed_abmil_seed{s}/abmil_best_model.pt') for s in SEEDS]),
        'bert': ('BERT-ABMIL', 'bert',
                 [ckpt('archive/bert_abmil/bert_abmil_best_model.pt')]),
        'sparse_bert': ('Sparse BERT-ABMIL', 'bert_sparse',
                        [ckpt('sparse_bert/sparse_bert_best_model.pt')]),
        'ai2tale': ('AI2TALE', 'ai2tale',
                    [ckpt('archive/ai2tale/ai2tale_best_model.pt')]),
        'snorkel': ('Snorkel', 'snorkel',
                    [ckpt('archive/snorkel/snorkel_best_model.pt')]),
    }
    wanted = [m.strip() for m in args.models.split(',')]

    # ---------------- stage 1: extract ----------------
    all_scores = {}
    if os.path.exists(args.scores_file):
        with open(args.scores_file) as f:
            all_scores = json.load(f)
        if not args.scores_only:
            print(f"loaded cached scores for: {list(all_scores.keys())}")

    if not args.scores_only:
        if args.device == 'auto':
            device = torch.device('mps' if torch.backends.mps.is_available()
                                  else 'cpu')
        else:
            device = torch.device(args.device)
        print(f"Device: {device}")
        config = load_config()
        _, test_loader = create_dataloaders(config)

        for key in wanted:
            if key == 'rules':
                name = 'Rules-only'
                if name in all_scores and not args.scores_only \
                        and len(all_scores[name]) >= len(wanted_ids):
                    print(f"[{name}] cached, skipping extraction")
                    continue
                print(f"[{name}] extracting ...")
                all_scores[name] = extract_scores_rules(test_loader, config,
                                                        wanted_ids)
            elif key == 'random':
                name = 'Random'
                if name in all_scores and len(all_scores[name]) >= len(wanted_ids):
                    print(f"[{name}] cached, skipping extraction")
                    continue
                print(f"[{name}] generating (seed 0) ...")
                all_scores[name] = extract_scores_random(test_loader, wanted_ids)
            elif key in registry:
                name, kind, paths = registry[key]
                if name in all_scores and len(all_scores[name]) >= len(wanted_ids) \
                        and len(paths) == 1:
                    print(f"[{name}] cached, skipping extraction")
                    continue
                for pi, path in enumerate(paths):
                    tag = name if len(paths) == 1 else f"{name} [seed{SEEDS[pi]}]"
                    if tag in all_scores and len(all_scores[tag]) >= len(wanted_ids):
                        print(f"[{tag}] cached, skipping")
                        continue
                    print(f"[{tag}] extracting from {os.path.basename(path)} ...")
                    model = build_model(kind)
                    model.load_state_dict(load_state(path), strict=False)
                    model.to(device)
                    model.eval()
                    if kind == 'snorkel':
                        all_scores[tag] = extract_scores_snorkel(
                            model, test_loader, device, wanted_ids)
                    else:
                        all_scores[tag] = extract_scores_model(
                            model, kind, test_loader, device, wanted_ids)
                    del model
            else:
                print(f"unknown model key: {key}")

        with open(args.scores_file, 'w') as f:
            json.dump(all_scores, f)
        print(f"saved scores -> {args.scores_file} "
              f"({len(all_scores)} model entries)")

    if args.extract_only:
        return

    # ---------------- stage 2: score ----------------
    results = {}
    per_email_store = {}
    for name, sc in all_scores.items():
        scores = {int(k): np.array(v) for k, v in sc.items()}
        res = evaluate_scores(scores, gold, ks=ks, mode=args.mode)
        per_email_store[name] = res.pop('_per_email')
        results[name] = res

    # paired differences vs reference
    # if reference is a multi-seed family name, build a seed-averaged pseudo
    # per-email reference so paired CIs are well defined
    if args.reference not in per_email_store:
        variants = [n for n in per_email_store
                    if n.startswith(args.reference + ' [seed')]
        if variants:
            common = set.intersection(*[set(per_email_store[v]) for v in variants])
            ref_avg = {}
            for eid in common:
                rows = [per_email_store[v][eid] for v in variants]
                entry = {'n_gold': rows[0].get('n_gold', 0)}
                for mkey in rows[0]:
                    if mkey == 'n_gold':
                        continue
                    vals = [r[mkey] for r in rows if mkey in r]
                    if vals:
                        entry[mkey] = float(np.mean(vals))
                ref_avg[eid] = entry
            per_email_store[args.reference] = ref_avg
            print(f"reference '{args.reference}': seed-averaged over "
                  f"{len(variants)} variants")
    if args.reference in per_email_store:
        ref_pe = per_email_store[args.reference]
        for name in results:
            if name == args.reference:
                continue
            diffs = {}
            for mkey in [f'p@{k}' for k in ks] + [f'r@{k}' for k in ks]:
                d = paired_diff_ci(per_email_store[name], ref_pe, mkey)
                if d is not None:
                    diffs[mkey] = d
            results[name]['paired_vs_reference'] = diffs

    # per-stratum breakdown
    if args.by_stratum:
        strat = gold['strata']
        by = {}
        for s in sorted(set(strat.values())):
            eids = {eid for eid, st in strat.items() if st == s}
            sub_labels = {k: v for k, v in gold['labels'].items() if k[0] in eids}
            sub_gold = {'labels': sub_labels, 'strata': strat,
                        'n_raters': gold['n_raters'], 'n_ties': 0,
                        'n_items': len(sub_labels)}
            by[s] = {}
            for name, sc in all_scores.items():
                scores = {int(k): np.array(v) for k, v in sc.items()}
                r = evaluate_scores(scores, sub_gold, ks=ks, mode=args.mode,
                                    bootstrap=False)
                r.pop('_per_email', None)
                by[s][name] = r
        results_by_stratum = by
    else:
        results_by_stratum = None

    print(f"\n=== Ground-truth localization ({args.mode}) ===")
    print(format_table(results, ks))

    payload = {
        'gold_meta': {'n_items': gold['n_items'], 'n_raters': gold['n_raters'],
                      'n_ties': gold['n_ties'], 'n_emails': len(wanted_ids)},
        'mode': args.mode, 'ks': list(ks),
        'reference': args.reference,
        'results': results,
    }
    if results_by_stratum is not None:
        payload['by_stratum'] = results_by_stratum
    with open(args.output, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {args.output}")


if __name__ == '__main__':
    main()
