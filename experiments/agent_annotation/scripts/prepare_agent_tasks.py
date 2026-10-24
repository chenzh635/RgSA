"""Prepare blinded task files for LLM-agent annotation.

Reads:
  experiments/human_annotation/pilot_30.json          (30-email pilot, has stratum)
  experiments/human_annotation/annotation_data.json   (150-email production, has model fields)

Writes (all under experiments/agent_annotation/tasks/):
  pilot_30_blind.json           — pilot emails stripped of `stratum`
  pilot_30_blind_shuffled.json  — same emails, sentence DISPLAY order shuffled
                                  per email (seeded; original `idx` kept).
                                  For the optional position-bias diagnostic
                                  pass (spec §6): annotate this file as
                                  `agent_pass_a_shuffled`, then run
                                  scripts/score_position_bias.py.
  production_chunk_{1..6}.json  — 150 emails stripped of pred_prob/correct/
                                  spearman/entropy/stratum/attn/rule_score,
                                  split into 6 chunks of 25
  _key_stratum.json             — sidecar: email_idx -> stratum (NEVER give to
                                  the annotation agent; used by scoring to
                                  re-attach strata for stratified analysis)

Blinding rationale: stratum encodes model behavior (spearman quartile /
misclassified); attn and rule_score are the very quantities under evaluation.
Leaking any of them would make the agent's labels circular.

Usage: python3 experiments/agent_annotation/scripts/prepare_agent_tasks.py
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
HA = os.path.join(ROOT, 'experiments', 'human_annotation')
OUT = os.path.join(HERE, '..', 'tasks')

EMAIL_STRIP = {'stratum', 'pred_prob', 'correct', 'spearman', 'entropy'}
SENT_STRIP = {'attn', 'rule_score'}


def blind_email(e):
    out = {k: v for k, v in e.items() if k not in EMAIL_STRIP}
    out['sentences'] = [
        {k: v for k, v in s.items() if k not in SENT_STRIP}
        for s in e['sentences']
    ]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    key = {}  # email_idx -> {'batch': ..., 'stratum': ...}

    # --- pilot ---
    with open(os.path.join(HA, 'pilot_30.json')) as f:
        pilot = json.load(f)
    blind_pilot = {
        'metadata': {
            'batch': 'pilot_30_blind',
            'n': len(pilot['emails']),
            'note': 'Blinded copy of pilot_30.json. Stratum removed. '
                    'Sentences: idx/text/zh (zh = reference translation only).',
        },
        'emails': [blind_email(e) for e in pilot['emails']],
    }
    for e in pilot['emails']:
        key[str(e['email_idx'])] = {'batch': 'pilot_30', 'stratum': e['stratum']}
    with open(os.path.join(OUT, 'pilot_30_blind.json'), 'w') as f:
        json.dump(blind_pilot, f, ensure_ascii=False, indent=1)
    print(f"pilot_30_blind.json: {len(blind_pilot['emails'])} emails")

    # --- pilot, shuffled display order (position-bias diagnostic, spec §6) ---
    rng = random.Random(0)
    shuf = json.loads(json.dumps(blind_pilot))  # deep copy
    shuf['metadata']['batch'] = 'pilot_30_blind_shuffled'
    shuf['metadata']['note'] = ('Same emails as pilot_30_blind; sentence display '
                                'order shuffled per email (seed 0). Original idx '
                                'preserved — annotate by idx. Position-bias '
                                'diagnostic only.')
    for e in shuf['emails']:
        sents = e['sentences'][:]
        rng.shuffle(sents)
        e['sentences'] = sents
    with open(os.path.join(OUT, 'pilot_30_blind_shuffled.json'), 'w') as f:
        json.dump(shuf, f, ensure_ascii=False, indent=1)
    print('pilot_30_blind_shuffled.json: display order shuffled (idx preserved)')

    # --- production ---
    with open(os.path.join(HA, 'annotation_data.json')) as f:
        prod = json.load(f)
    emails = prod['emails']
    for e in emails:
        key[str(e['email_idx'])] = {'batch': 'production_150', 'stratum': e['stratum']}
    chunk_size = 25
    n_chunks = (len(emails) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        chunk = emails[i * chunk_size:(i + 1) * chunk_size]
        doc = {
            'metadata': {
                'batch': f'production_chunk_{i+1}',
                'n': len(chunk),
                'note': 'Blinded production emails. All model-derived fields '
                        'removed. Sentences: idx/text only.',
            },
            'emails': [blind_email(e) for e in chunk],
        }
        path = os.path.join(OUT, f'production_chunk_{i+1}.json')
        with open(path, 'w') as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"production_chunk_{i+1}.json: {len(chunk)} emails")

    with open(os.path.join(OUT, '_key_stratum.json'), 'w') as f:
        json.dump(key, f, indent=1)
    print(f"_key_stratum.json: {len(key)} entries (sidecar — do NOT give to agent)")

    # sanity: no leaked fields anywhere
    for name in os.listdir(OUT):
        if name.startswith('_') or not name.endswith('.json'):
            continue
        with open(os.path.join(OUT, name)) as f:
            txt = f.read()
        for bad in ('stratum', 'pred_prob', 'spearman', 'entropy', 'rule_score', '"attn"'):
            assert bad not in txt, f'LEAK {bad} in {name}'
    print('blinding sanity check: OK')


if __name__ == '__main__':
    main()
