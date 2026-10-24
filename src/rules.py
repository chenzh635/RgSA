"""Domain knowledge labeling functions for phishing sentence scoring."""

import json
import re

import numpy as np


def load_keywords(keywords_path):
    """Load cognitive trigger keywords from JSON file."""
    with open(keywords_path, 'r') as f:
        return json.load(f)


def score_sentences(raw_sentences, keywords):
    """Score each sentence using domain knowledge rules.

    Args:
        raw_sentences: list of sentence strings
        keywords: dict of {category: [word_stems]} from kwords_stem.json

    Returns:
        rule_score: np.array of shape (max_sentences,) with values in [0, 1]
    """
    scores = np.zeros(len(raw_sentences), dtype=np.float32)

    for i, sent in enumerate(raw_sentences):
        sent_lower = sent.lower()
        triggers = 0
        total_checks = 0

        # Check cognitive trigger categories (excluding english_words)
        for cat, stems in keywords.items():
            if cat == 'english_words':
                continue
            total_checks += 1
            # Stem matching: check if any stem appears in the sentence
            for stem in stems:
                if stem in sent_lower:
                    triggers += 1
                    break  # One match per category is enough

        # Rule-based signals
        has_uri = bool(re.search(r'https?://|www\.|click\s+here|link\s+below', sent_lower))
        has_generic_greeting = bool(re.search(r'dear\s+(customer|user|member|beneficiary|sir)', sent_lower))
        has_action = bool(re.search(r'click\s+here|update\s+your|confirm\s+your|verify\s+your|renew', sent_lower))

        # Aggregate score: normalize to [0, 1]
        rule_score = (triggers / max(total_checks, 1) + has_uri * 0.3 +
                      has_generic_greeting * 0.2 + has_action * 0.2)
        scores[i] = min(rule_score, 1.0)

    return scores


def batch_score(raw_sentences_batch, keywords):
    """Score sentences for a batch of emails.

    Args:
        raw_sentences_batch: list of list of strings (batch_size x variable_sentences)
        keywords: dict from kwords_stem.json

    Returns:
        rule_scores: np.array of shape (batch_size, max_sentences_in_batch)
    """
    batch_size = len(raw_sentences_batch)
    max_sents = max(len(sents) for sents in raw_sentences_batch)
    all_scores = np.zeros((batch_size, max_sents), dtype=np.float32)

    for b, sentences in enumerate(raw_sentences_batch):
        all_scores[b, :len(sentences)] = score_sentences(sentences, keywords)

    return all_scores
