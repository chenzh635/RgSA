"""Score agent-annotation passes against human pilot annotations (calibration gate).

Usage:
  python3 experiments/agent_annotation/scripts/score_agent_calibration.py \
      outputs/agent_pass_a_pilot_30_blind.json \
      outputs/agent_pass_b_pilot_30_blind.json \
      outputs/agent_pass_c_pilot_30_blind.json

Compares the (typically 3) blinded agent passes on pilot_30 against the two
human raters (chan, li) using src.annotation_agreement.

Gates (all must hold to proceed to production; rebased after calibration
round 1 — see CALIBRATION_LOG.md §B: with a human-human ceiling of κ=0.398,
the spec's 0.60-vs-noisy-anchor target is statistically unreachable):
  G1   each agent pass: mean κ(agent, {chan, li}) >= 0.50
  G1b  parity — for each human h: κ(agent, h) >= κ(other_human, h)
       (the agent must agree with each human at least as well as the other
        human does — "at least a competent human rater")
  G2   each agent pair: κ(pass_i, pass_j)        >= 0.60  (spec target: 0.7)
  G3   each agent pass: mean κ(agent, human)     >= κ(chan, li)  (recomputed)

Also reports (spec §7/§8):
  - email-level agreement agent vs each human: Y-set Jaccard, exact match,
    first-Y agreement, P@1-style hit rate (agent's first Y ∈ human Y set)
  - oracle coverage: fraction of emails with >= 1 Y per rater
  - low-confidence share per agent pass

Sensitivity: all gates re-checked with email 4860 excluded (dominant source of
human-human disagreement in the pilot, 11/52 items).

Writes outputs/calibration_report.json; prints a PASS/FAIL verdict.
Strata are re-attached from tasks/_key_stratum.json (agents never see them).
"""

import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # project root
sys.path.insert(0, str(ROOT))

from src.annotation_agreement import analyze, load_annotations  # noqa: E402

HA = ROOT / 'experiments' / 'human_annotation'
HUMAN_EXPORTS = [
    HA / 'exports' / 'annotations_chan_pilot_30_2026-09-01-15-13.json',
    HA / 'exports' / 'annotations_li_pilot_30_2026-09-06-14-45.json',
]
HUMAN_NAMES = ['chan', 'li']
KEY_FILE = HERE.parent / 'tasks' / '_key_stratum.json'
OUT_FILE = HERE.parent / 'outputs' / 'calibration_report.json'

G1_MIN = 0.50   # rebased from 0.60 (round 1, CALIBRATION_LOG.md §B)
G2_MIN = 0.60
SENSITIVITY_EMAIL = 4860


def attach_stratum(ann, key):
    for (email_id, _), v in ann.items():
        entry = key.get(str(email_id))
        v['stratum'] = entry['stratum'] if entry else ''
    return ann


def subset_excluding(ann, email_id):
    return {k: v for k, v in ann.items() if k[0] != email_id}


def pairwise_kappa(ann_a, ann_b):
    common = sorted(set(ann_a) & set(ann_b))
    from src.annotation_agreement import cohens_kappa
    return cohens_kappa([ann_a[k]['label'] for k in common],
                        [ann_b[k]['label'] for k in common])


def y_sets(ann):
    """email_id -> set of sentence_idx labeled Y."""
    out = {}
    for (eid, sidx), v in ann.items():
        out.setdefault(eid, set())
        if v['label'] == 'Y':
            out[eid].add(sidx)
    return out


def email_level_metrics(ann_a, ann_b):
    """Email-level agreement between two raters (a=agent, b=human)."""
    ya, yb = y_sets(ann_a), y_sets(ann_b)
    emails = sorted(set(ya) | set(yb))
    jac, exact, firsty, p1_hit, p1_den = [], 0, [], 0, 0
    for e in emails:
        sa, sb = ya.get(e, set()), yb.get(e, set())
        union = sa | sb
        jac.append(len(sa & sb) / len(union) if union else 1.0)
        exact += int(sa == sb)
        if sa and sb:
            firsty.append(int(min(sa) == min(sb)))
        if sb:  # human says email has evidence; does agent's first Y hit it?
            p1_den += 1
            p1_hit += int(bool(sa) and min(sa) in sb)
    return {
        'n_emails': len(emails),
        'yset_jaccard_mean': sum(jac) / len(jac) if jac else None,
        'yset_exact_match': exact / len(emails) if emails else None,
        'first_y_agreement': sum(firsty) / len(firsty) if firsty else None,
        'p_at_1_agent_first_y_in_human_yset': (p1_hit / p1_den) if p1_den else None,
        'p_at_1_denominator': p1_den,
    }


def coverage(ann):
    ys = y_sets(ann)
    emails = {k[0] for k in ann}
    nonempty = sum(1 for e in emails if ys.get(e))
    return {'n_emails': len(emails),
            'emails_with_ge1_Y': nonempty,
            'oracle_coverage': nonempty / len(emails) if emails else None}


def confidence_stats(path):
    """Share of low-confidence labels in a pass file (JSON export format)."""
    with open(path) as f:
        data = json.load(f)
    rows = [r for r in data.get('annotations', []) if r.get('label') in ('Y', 'N', '?')]
    if not rows:
        return None
    conf = Counter(str(r.get('confidence', 'unspecified')) for r in rows)
    return {'n': len(rows), 'by_confidence': dict(conf),
            'low_share': conf.get('low', 0) / len(rows)}


def main():
    pass_paths = [Path(p) for p in sys.argv[1:]]
    if len(pass_paths) < 2:
        print(__doc__)
        sys.exit(1)

    with open(KEY_FILE) as f:
        key = json.load(f)

    humans = [attach_stratum(load_annotations(p), key) for p in HUMAN_EXPORTS]
    agents, agent_names = [], []
    for p in pass_paths:
        ann = attach_stratum(load_annotations(p), key)
        agents.append(ann)
        agent_names.append(p.stem.split('_pilot')[0])  # e.g. agent_pass_a

    # ---- coverage sanity ----
    ref_keys = set(humans[0])
    for name, ann in zip(agent_names, agents):
        missing = ref_keys - set(ann)
        extra = set(ann) - ref_keys
        if missing or extra:
            print(f'WARNING {name}: missing {len(missing)} items, extra {len(extra)} items')

    report = {'passes': agent_names, 'gates': {}, 'kappa': {}}

    # ---- full 5-rater analysis (fleiss, pairwise, by-stratum, disagreements) ----
    full = analyze(humans + agents, HUMAN_NAMES + agent_names)
    report['n_items'] = full['n_items']
    report['items_dropped_not_common'] = full['items_dropped_not_common']
    report['fleiss_kappa_5rater'] = full['fleiss_kappa']
    report['pairwise'] = full['pairwise']
    report['by_stratum'] = full['by_stratum']
    report['n_disagreements'] = full['n_disagreements']
    report['pilot_disagreements'] = full['_disagreements']  # for protocol revision

    pw = full['pairwise']
    k_chan_li = pw['chan~li']['cohens_kappa']
    report['kappa']['chan~li'] = k_chan_li

    # ---- gates on the full pilot ----
    def gate_checks(human_anns, agent_anns, tag, baseline):
        g = {'G1': {}, 'G1b': {}, 'G2': {}, 'G3': {}}
        for name, ann in zip(agent_names, agent_anns):
            ks = {}
            for hname, hann in zip(HUMAN_NAMES, human_anns):
                k = pairwise_kappa(ann, hann)
                ks[hname] = k
                report['kappa'][f'{tag}{name}~{hname}'] = k
            mean_k = sum(ks.values()) / len(ks)
            g['G1'][name] = {'mean_kappa_vs_humans': mean_k,
                             'pass': mean_k >= G1_MIN}
            # parity: with 2 humans, the "other human" baseline for each h is
            # κ(chan,li) itself — the agent must beat it against BOTH humans
            g['G1b'][name] = {f'kappa_vs_{h}': k for h, k in ks.items()}
            g['G1b'][name]['baseline'] = baseline
            g['G1b'][name]['pass'] = all(k >= baseline for k in ks.values())
            g['G3'][name] = {'mean_kappa_vs_humans': mean_k,
                             'kappa_chan_li': baseline,
                             'pass': mean_k >= baseline}
        for (n1, a1), (n2, a2) in itertools.combinations(
                list(zip(agent_names, agent_anns)), 2):
            k = pairwise_kappa(a1, a2)
            report['kappa'][f'{tag}{n1}~{n2}'] = k
            g['G2'][f'{n1}~{n2}'] = {'kappa': k, 'pass': k >= G2_MIN}
        return g

    report['gates']['full'] = gate_checks(humans, agents, '', k_chan_li)

    # ---- sensitivity: exclude the dominant-disagreement email ----
    humans_s = [subset_excluding(h, SENSITIVITY_EMAIL) for h in humans]
    agents_s = [subset_excluding(a, SENSITIVITY_EMAIL) for a in agents]
    k_chan_li_s = pairwise_kappa(humans_s[0], humans_s[1])
    report['kappa'][f'excl{SENSITIVITY_EMAIL}_chan~li'] = k_chan_li_s
    sens = gate_checks(humans_s, agents_s, f'excl{SENSITIVITY_EMAIL}_',
                       k_chan_li_s)
    report['gates'][f'excl_{SENSITIVITY_EMAIL}'] = sens

    # ---- email-level agreement, coverage, confidence (spec §7/§8) ----
    report['email_level'] = {}
    for name, ann in zip(agent_names, agents):
        report['email_level'][name] = {
            f'vs_{hname}': email_level_metrics(ann, hann)
            for hname, hann in zip(HUMAN_NAMES, humans)
        }
    report['coverage'] = {n: coverage(a) for n, a in
                          list(zip(HUMAN_NAMES, humans)) +
                          list(zip(agent_names, agents))}
    report['confidence'] = {n: confidence_stats(p)
                            for n, p in zip(agent_names, pass_paths)}

    # ---- verdict ----
    def all_pass(g):
        return all(v['pass'] for grp in g.values() for v in grp.values())

    verdict_full = all_pass(report['gates']['full'])
    verdict_sens = all_pass(report['gates'][f'excl_{SENSITIVITY_EMAIL}'])
    report['verdict'] = {
        'full': 'PASS' if verdict_full else 'FAIL',
        f'excl_{SENSITIVITY_EMAIL}': 'PASS' if verdict_sens else 'FAIL',
        'decision': ('PASS — proceed to production' if verdict_full else
                     ('MARGINAL — passes only with email 4860 excluded; '
                      'revise protocol/判例 then re-run pilot' if verdict_sens else
                      'FAIL — do NOT run production; revise protocol/prompts '
                      'and re-run the 3 passes')),
    }

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)

    # ---- console summary ----
    print(f"\nraters: {HUMAN_NAMES + agent_names}  |  items: {report['n_items']}"
          f"  |  Fleiss κ (5-rater): {report['fleiss_kappa_5rater']:.3f}")
    print(f"κ(chan,li) baseline: {k_chan_li:.3f} "
          f"(excl-{SENSITIVITY_EMAIL}: {k_chan_li_s:.3f})")
    for name in agent_names:
        g1 = report['gates']['full']['G1'][name]['mean_kappa_vs_humans']
        print(f"  {name}: mean κ vs humans = {g1:.3f}")
    for pair, v in report['gates']['full']['G2'].items():
        print(f"  inter-agent κ {pair} = {v['kappa']:.3f}")
    print(f"\nVERDICT (full pilot): {report['verdict']['full']}")
    print(f"VERDICT (excl {SENSITIVITY_EMAIL}): "
          f"{report['verdict'][f'excl_{SENSITIVITY_EMAIL}']}")
    print(f"=> {report['verdict']['decision']}")
    print(f"report -> {OUT_FILE}")


if __name__ == '__main__':
    main()
