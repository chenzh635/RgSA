"""铁律 2 审计：确保协议/prompt 中没有"关键词答案册"。

Scans protocol + prompt files (the only text an annotation agent is pointed
at) for leakage of the rule prior's answer key:
  1. Cialdini / trigger-category taxonomy names used by kwords_stem.json
  2. literal trigger phrases (the kind that appear in the keyword lists)
  3. density of distinct keyword stems from kwords_stem.json (categories only,
     english_words excluded) appearing as whole words — a handful of common
     English words is unavoidable and fine; a high density means the document
     is effectively reproducing the word list.

Exit code 0 = clean; 1 = leakage found (fix the documents before launching
any annotation pass).

Usage: python3 experiments/agent_annotation/scripts/check_prompt_leakage.py
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
BASE = os.path.join(HERE, '..')

DOCS = [
    os.path.join(BASE, 'protocol', 'PROTOCOL_AGENT_v1.md'),
    os.path.join(BASE, 'prompts', 'annotator_prompt.md'),
    os.path.join(BASE, 'prompts', 'adjudicator_prompt.md'),
]

CATEGORY_NAMES = ['cialdini', 'scarcity', 'reciprocity', 'socialproof',
                  'social proof', 'liking', 'consistency']  # taxonomy leakage

TRIGGER_PHRASES = [  # literal answer-key phrases (v1.0 examples, now removed)
    'verify your', 'click here', 'click the link', 'account will be suspended',
    'account suspended', 'atm pin', 'dear valued customer', 'urgent:',
    'your account', 'full name and', 'security team of',
]

DENSITY_LIMIT = 8  # distinct keyword stems allowed before flagging

# Stems that are unavoidable task meta-language (describing the annotation task
# itself: "phishing email", "human protocol", "review", JSON field examples),
# NOT trigger vocabulary. Verified by manual audit 2026-09-12.
META_WHITELIST = {'1', 'agre', 'author', 'claim', 'email', 'experi', 'go',
                  'hi', 'human', 'phish', 'product', 'protocol', 'review'}


def main():
    with open(os.path.join(ROOT, 'phishingdata', 'kwords_stem.json')) as f:
        kw = json.load(f)
    stems = set()
    for cat, lst in kw.items():
        if cat == 'english_words':
            continue
        stems.update(w.lower() for w in lst)

    bad = False
    for doc in DOCS:
        with open(doc) as f:
            text = f.read()
        low = text.lower()
        name = os.path.relpath(doc, ROOT)

        hits_cat = [c for c in CATEGORY_NAMES if c in low]
        hits_phrase = [p for p in TRIGGER_PHRASES if p in low]
        # whole-word stem matches (stems are Porter-truncated, e.g. 'expir')
        hits_stem = sorted({s for s in stems if s not in META_WHITELIST
                            and re.search(r'(?<![a-z])' + re.escape(s), low)})
        status = 'CLEAN'
        if hits_cat or hits_phrase or len(hits_stem) > DENSITY_LIMIT:
            status = 'LEAK'
            bad = True
        print(f'[{status}] {name}')
        if hits_cat:
            print(f'   category names: {hits_cat}')
        if hits_phrase:
            print(f'   trigger phrases: {hits_phrase}')
        print(f'   distinct keyword stems present: {len(hits_stem)} '
              f'(limit {DENSITY_LIMIT})'
              + (f' -> {hits_stem[:20]}' if len(hits_stem) > DENSITY_LIMIT else ''))

    # also verify no doc points the agent at the answer key as readable input
    for doc in DOCS:
        with open(doc) as f:
            text = f.read()
        if 'kwords' in text and '禁止' not in text and 'NEVER' not in text.upper():
            print(f'[LEAK] {doc} mentions kwords without a prohibition')
            bad = True

    print('\nVERDICT:', 'FAIL — fix docs before launching passes' if bad else 'PASS')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
