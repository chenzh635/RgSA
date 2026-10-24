"""Evaluate ERASER-style faithfulness metrics (Comprehensiveness/Sufficiency)
for all 7 models in paper Table 2.

Usage:
    python experiments/scripts/eval_faithfulness.py [--models sparse,abmil,...] \
        [--ks 1,3,5] [--device mps|cpu]

Output: experiments/faithfulness_results.json + printed table.

Note on CNN-model seeds: following the paper convention, ABMIL and
Sparse ABMIL report mean +/- std over seeds 123/456/789; BERT variants,
AI2TALE, Snorkel and Rules-only are single-run.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.baselines.ai2tale_pytorch import AI2TALEModel
from src.baselines.bert_abmil import BertABMIL
from src.baselines.snorkel_baseline import SentenceClassifier
from src.dataset import create_dataloaders
from src.evaluate_faithfulness import (
    evaluate_faithfulness_pooling,
    evaluate_faithfulness_rules_ref,
    evaluate_faithfulness_snorkel,
)
from src.models.abmil import ABMILBaseline, SparseABMIL

SEEDS = [123, 456, 789]


def load_config():
    with open(os.path.join(project_root, 'experiments/configs/defaults.yaml')) as f:
        return yaml.safe_load(f)


def load_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def load_sparse_bert_class():
    path = os.path.join(project_root, 'experiments/scripts/run_sparse_bert.py')
    spec = importlib.util.spec_from_file_location('run_sparse_bert', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SparseBertABMIL


def ckpt(rel):
    return os.path.join(project_root, 'experiments', rel)


def eval_one(name, kind, model, test_loader, device, ks, classify_attr='classifier'):
    print(f"  evaluating {name} ...", flush=True)
    if kind == 'snorkel':
        return evaluate_faithfulness_snorkel(model, test_loader, device, ks=ks)
    return evaluate_faithfulness_pooling(
        model, kind, test_loader, device, ks=ks, classify_attr=classify_attr)


def build_model(kind):
    if kind == 'cnn_sparse':
        return SparseABMIL(hidden_dim=64, attention_hidden=64,
                           classifier_hidden=[128, 64], use_sparse=True)
    if kind == 'cnn_abmil':
        return ABMILBaseline(hidden_dim=64, attention_hidden=64,
                             classifier_hidden=[128, 64])
    if kind in ('bert', 'bert_sparse'):
        cls = load_sparse_bert_class() if kind == 'bert_sparse' else BertABMIL
        return cls(attention_hidden=64, classifier_hidden=[128, 64])
    if kind == 'ai2tale':
        return AI2TALEModel()
    if kind == 'snorkel':
        return SentenceClassifier()
    raise ValueError(kind)


def run_model_group(name, kind, ckpt_paths, test_loader, device, ks,
                    classify_attr='classifier'):
    """Evaluate one or more checkpoints; aggregate mean +/- std."""
    per_run = []
    for path in ckpt_paths:
        model = build_model(kind)
        model.load_state_dict(load_state(path), strict=False)
        model.to(device)
        model.eval()
        per_run.append(eval_one(name, kind, model, test_loader, device, ks,
                                classify_attr))
        del model
    return per_run


def fmt_mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '   -  '
    if len(vals) == 1:
        return f'{vals[0]:7.3f}'
    return f'{np.mean(vals):.3f}±{np.std(vals):.3f}'


def print_table(results, ks):
    print(f"\n{'='*110}")
    print("Faithfulness summary (logit space). Scope: correctly classified "
          "phishing emails (ERASER convention);")
    print("models with no correct predictions fall back to all phishing emails "
          "and are marked (*).")
    print(f"{'='*110}")
    for k in ks:
        print(f"\n--- k = {k} ---")
        print(f"{'Model':<26}{'Comp@k':>16}{'Rand@k':>16}{'Advantage':>16}"
              f"{'Comp-rel':>12}{'Suff keep-p':>14}{'n_eval':>8}")
        for name, entry in results['models'].items():
            runs = entry['runs']
            # per-run scope: correct subset if nonempty, else all phishing
            scopes = ['correct' if r['n_correct'] > 0 else 'all' for r in runs]
            comp, rand, adv, crel, suff = [], [], [], [], []
            for r, scope in zip(runs, scopes):
                pk = r['per_k'][str(k)]
                comp.append(pk[f'comp_logit_{scope}'])
                rand.append(pk[f'rand_comp_logit_{scope}'])
                adv.append(pk[f'comp_advantage_{scope}'])
                crel.append(pk.get(f'comp_rel_{scope}'))
                suff.append(pk[f'suff_keep_prob_{scope}'])
            n_eval = sum(r['n_correct'] if s == 'correct'
                         else r['n_phishing'] - r['n_short_excluded']
                         for r, s in zip(runs, scopes))
            mark = ' *' if any(s == 'all' for s in scopes) else ''
            print(f"{name:<26}{fmt_mean_std(comp):>16}{fmt_mean_std(rand):>16}"
                  f"{fmt_mean_std(adv):>16}{fmt_mean_std(crel):>12}"
                  f"{fmt_mean_std(suff):>14}{n_eval:>8}{mark}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str,
                        default='sparse,abmil,bert,sparse_bert,ai2tale,snorkel,rules')
    parser.add_argument('--ks', type=str, default='1,3,5')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str,
                        default=os.path.join(project_root, 'experiments',
                                             'faithfulness_results.json'))
    parser.add_argument('--resume', action='store_true',
                        help='Skip models already present in the output file')
    args = parser.parse_args()

    ks = tuple(int(x) for x in args.ks.split(','))
    wanted = [m.strip() for m in args.models.split(',')]

    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    config = load_config()
    _, test_loader = create_dataloaders(config)
    print(f"Test emails: {len(test_loader.dataset)}")

    results = {'meta': {
        'ks': list(ks), 'device': str(device),
        'protocol': ('ERASER-style perturbation; logit space; top-k by selection '
                     'weight; masked weights renormalized; scope = correctly '
                     'classified phishing emails; random-k control R=5, seed=0'),
        'rules_ref_model': 'ABMIL (multi_seed seed=123)',
    }, 'models': {}}

    def save():
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        results['models'] = prev.get('models', {})
        print(f"Resuming: {len(results['models'])} model group(s) already done")

    def done(name):
        return args.resume and name in results['models']

    registry = {
        'sparse': ('Sparse ABMIL (Ours)', 'cnn_sparse',
                   [ckpt(f'multi_seed_sparse_seed{s}/sparse_best_model.pt') for s in SEEDS],
                   'classifier'),
        'abmil': ('ABMIL (CNN)', 'cnn_abmil',
                  [ckpt(f'multi_seed_abmil_seed{s}/abmil_best_model.pt') for s in SEEDS],
                  'classifier'),
        'bert': ('BERT-ABMIL', 'bert',
                 [ckpt('archive/bert_abmil/bert_abmil_best_model.pt')], 'classifier'),
        'sparse_bert': ('Sparse BERT-ABMIL', 'bert_sparse',
                        [ckpt('sparse_bert/sparse_bert_best_model.pt')], 'classifier'),
        'ai2tale': ('AI2TALE', 'ai2tale',
                    [ckpt('archive/ai2tale/ai2tale_best_model.pt')], 'predictor'),
        'snorkel': ('Snorkel', 'snorkel',
                    [ckpt('archive/snorkel/snorkel_best_model.pt')], None),
    }

    for key in wanted:
        if key == 'rules':
            rname = 'Rules-only (ref: ABMIL)'
            if done(rname):
                print(f"\n[{rname}] already done, skipping")
                continue
            print("\n[Rules-only (selection) + ABMIL (decision)]")
            ref = build_model('cnn_abmil')
            ref.load_state_dict(load_state(
                ckpt('multi_seed_abmil_seed123/abmil_best_model.pt')), strict=False)
            ref.to(device)
            summary = evaluate_faithfulness_rules_ref(
                ref, test_loader, config, device, ks=ks)
            results['models'][rname] = {'runs': [summary]}
            del ref
            save()
        elif key in registry:
            name, kind, paths, attr = registry[key]
            if done(name):
                print(f"\n[{name}] already done, skipping")
                continue
            print(f"\n[{name}] ({len(paths)} run(s))")
            runs = run_model_group(name, kind, paths, test_loader, device, ks,
                                   classify_attr=attr)
            results['models'][name] = {'runs': runs}
            save()
        else:
            print(f"unknown model key: {key}")

    save()
    print(f"\nSaved: {args.output}")

    print_table(results, ks)


if __name__ == '__main__':
    main()
