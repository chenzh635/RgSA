"""位置偏差诊断（任务书 §6）：同一模型顺序版 vs 打乱版的标签翻转率。

Usage:
  python3 scripts/score_position_bias.py \
      outputs/agent_pass_a_pilot_30_blind.json \
      outputs/agent_pass_a_pilot_30_blind_shuffled.json

Reports per-item flip rate, Cohen's κ between the two orderings, and the
flip confusion matrix (e.g. Y->N vs N->Y asymmetry hints at primacy bias).
Diagnostic only — no gate. Write the conclusion into the agreement report.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.annotation_agreement import cohens_kappa, load_annotations, raw_agreement  # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ordered = load_annotations(Path(sys.argv[1]))
    shuffled = load_annotations(Path(sys.argv[2]))
    common = sorted(set(ordered) & set(shuffled))
    if len(common) < min(len(ordered), len(shuffled)):
        print(f'WARNING: coverage differs ({len(ordered)} vs {len(shuffled)}); '
              f'comparing {len(common)} common items')
    if not common:
        sys.exit('no common items')

    la = [ordered[k]['label'] for k in common]
    lb = [shuffled[k]['label'] for k in common]
    flips = [(k, ordered[k]['label'], shuffled[k]['label'])
             for k in common if ordered[k]['label'] != shuffled[k]['label']]
    conf = Counter(f'{a}>{b}' for _, a, b in flips)

    out = {
        'n_items': len(common),
        'raw_agreement': raw_agreement(la, lb),
        'cohens_kappa': cohens_kappa(la, lb),
        'flip_rate': len(flips) / len(common),
        'n_flips': len(flips),
        'flip_confusion': dict(conf),
        'flips': [{'email_id': k[0], 'sentence_idx': k[1],
                   'ordered': a, 'shuffled': b} for k, a, b in flips],
    }
    out_path = HERE.parent / 'outputs' / 'position_bias_report.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"items: {out['n_items']} | raw agreement: {out['raw_agreement']:.3f} "
          f"| κ(ordered,shuffled): {out['cohens_kappa']:.3f}")
    print(f"flip rate: {out['flip_rate']:.3f} ({out['n_flips']} items)")
    print(f"flip confusion: {out['flip_confusion']}")
    print(f"report -> {out_path}")
    if out['flip_rate'] > 0.10:
        print('NOTE: flip rate >10% — 顺序敏感明显，论文需披露位置偏差检查结果')


if __name__ == '__main__':
    main()
