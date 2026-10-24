"""Prepare human annotation dataset for phishing sentence localization evaluation.

Selects a stratified sample of 150 phishing emails covering:
- High Spearman (model agrees with rules)
- Low Spearman (model disagrees with rules)
- Misclassified emails
- Various email lengths

Outputs:
1. annotation_data.json — full data for analysis
2. annotation_sheet.csv — simple spreadsheet for human annotators
"""

import os
import sys
import json
import csv
import random
import numpy as np
import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.abmil import SparseABMIL
from src.dataset import create_dataloaders
from src.rules import batch_score
from scipy.stats import spearmanr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(project_root, 'experiments', 'configs', 'defaults.yaml'))
    parser.add_argument('--checkpoint', default=os.path.join(project_root, 'experiments', 'archive', 'sparse_best_model.pt'))
    parser.add_argument('--output-dir', default=os.path.join(project_root, 'experiments', 'human_annotation'))
    parser.add_argument('--n-samples', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config and model
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = SparseABMIL(
        vocab_size=config.get('vocab_size', 311774),
        embed_dim=config.get('embed_dim', 150),
        hidden_dim=config.get('hidden_dim', 150),
        attention_hidden=config.get('attention_hidden', 64),
        use_sparse=True,
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Load test data
    _, test_loader = create_dataloaders(config)
    print(f"Test set: {len(test_loader.dataset)} emails")

    # Load keywords
    keywords_path = os.path.join(project_root, 'phishingdata', 'kwords_stem.json')
    with open(keywords_path) as f:
        keywords = json.load(f)

    # Collect all phishing emails with metrics
    all_emails = []

    with torch.no_grad():
        email_idx = 0
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)
            sent_lengths = batch['sent_lengths'].to(device)
            raw_sentences = batch['raw_sentences']

            output, attn_weights, attn_scores, _ = model(tokens, sent_lengths, temperature=0.1)
            pred_probs = torch.sigmoid(output).cpu().numpy()
            rule_scores = batch_score(raw_sentences, keywords)

            B, S = attn_weights.shape
            for b in range(B):
                if labels[b].item() != 1:  # Only phishing emails
                    email_idx += 1
                    continue

                pred_prob = float(pred_probs[b])
                pred = 1 if pred_prob > 0.5 else 0
                n_valid = int((sent_lengths[b] > 0).sum().item())

                if n_valid < 2 or len(raw_sentences[b]) < 2:
                    email_idx += 1
                    continue

                attn_b = attn_weights[b, :n_valid].cpu().numpy()
                n_rule = rule_scores.shape[1]
                n_use = min(n_valid, n_rule)
                rule_b = rule_scores[b, :n_use]

                # Compute Spearman
                spearman_val = None
                if n_use >= 2 and len(set(attn_b[:n_use])) >= 2 and len(set(rule_b)) >= 2:
                    corr, _ = spearmanr(attn_b[:n_use], rule_b)
                    if not np.isnan(corr):
                        spearman_val = float(corr)

                # Compute entropy
                p = attn_b / (attn_b.sum() + 1e-10)
                entropy = float(-np.sum(p * np.log(p + 1e-10)))

                # Build sentence data
                n_raw = len(raw_sentences[b])
                n_use_sent = min(n_valid, n_raw)
                sentences = []
                for i in range(n_use_sent):
                    rule_s = float(rule_b[i]) if i < len(rule_b) else 0.0
                    sentences.append({
                        'idx': i,
                        'text': raw_sentences[b][i],
                        'attn': float(attn_b[i]),
                        'rule_score': rule_s,
                    })

                all_emails.append({
                    'email_idx': email_idx,
                    'pred_prob': pred_prob,
                    'correct': pred == 1,
                    'n_sentences': len(sentences),
                    'spearman': spearman_val,
                    'entropy': entropy,
                    'sentences': sentences,
                })
                email_idx += 1

    print(f"Collected {len(all_emails)} phishing emails with ≥2 sentences")

    # Stratified sampling
    # Categorize by Spearman quartile
    spearman_values = sorted([e['spearman'] for e in all_emails if e['spearman'] is not None])
    q25 = spearman_values[len(spearman_values) // 4]
    q50 = spearman_values[len(spearman_values) // 2]
    q75 = spearman_values[3 * len(spearman_values) // 4]

    strata = {
        'misclassified': [e for e in all_emails if not e['correct']],
        'low_spearman': [e for e in all_emails if e['spearman'] is not None and e['spearman'] < q25],
        'mid_spearman': [e for e in all_emails if e['spearman'] is not None and q25 <= e['spearman'] < q75],
        'high_spearman': [e for e in all_emails if e['spearman'] is not None and e['spearman'] >= q75],
    }

    print(f"\nStrata sizes: misclassified={len(strata['misclassified'])}, "
          f"low={len(strata['low_spearman'])}, mid={len(strata['mid_spearman'])}, "
          f"high={len(strata['high_spearman'])}")
    print(f"Spearman quartiles: Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}")

    # Sample from each stratum
    n = args.n_samples
    allocation = {
        'misclassified': min(len(strata['misclassified']), n // 10),      # ~15
        'low_spearman': n // 3,                                            # ~50
        'mid_spearman': n // 3,                                            # ~50
        'high_spearman': n - n // 10 - 2 * (n // 3),                      # ~35
    }

    selected = []
    seen_ids = set()
    for stratum_name, stratum_count in allocation.items():
        candidates = strata[stratum_name]
        random.shuffle(candidates)
        for e in candidates:
            if e['email_idx'] not in seen_ids and len([s for s in selected if s.get('stratum') == stratum_name]) < stratum_count:
                e['stratum'] = stratum_name
                selected.append(e)
                seen_ids.add(e['email_idx'])

    # Shuffle final selection
    random.shuffle(selected)
    print(f"\nSelected {len(selected)} emails for annotation")

    # Save full annotation data
    annotation_data_path = os.path.join(args.output_dir, 'annotation_data.json')
    with open(annotation_data_path, 'w') as f:
        json.dump({
            'metadata': {
                'total_phishing': len(all_emails),
                'selected': len(selected),
                'allocation': allocation,
                'quartiles': {'q25': q25, 'q50': q50, 'q75': q75},
            },
            'emails': selected,
        }, f, indent=2)
    print(f"Full data saved to: {annotation_data_path}")

    # Generate annotation sheet (CSV for human annotators)
    csv_path = os.path.join(args.output_dir, 'annotation_sheet.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'email_id', 'stratum', 'sentence_idx', 'sentence_text',
            'model_attention', 'rule_score',
            'is_phishing_sentence (Y/N/?)'
        ])

        for email in selected:
            # Sort sentences by attention (descending) for annotation
            sorted_sents = sorted(email['sentences'], key=lambda s: s['attn'], reverse=True)
            for sent in sorted_sents:
                writer.writerow([
                    email['email_idx'],
                    email['stratum'],
                    sent['idx'],
                    sent['text'][:200],
                    f"{sent['attn']:.4f}",
                    f"{sent['rule_score']:.2f}",
                    '',  # Empty for annotator
                ])
            # Add blank row between emails
            writer.writerow([''] * 7)

    print(f"Annotation sheet saved to: {csv_path}")

    # Generate annotation protocol
    protocol_path = os.path.join(args.output_dir, 'ANNOTATION_PROTOCOL.md')
    with open(protocol_path, 'w') as f:
        f.write("""# Phishing Sentence Localization — Human Annotation Protocol

## Task

For each email in the annotation sheet, identify which sentence(s) you believe make the email a phishing email.

## Instructions

1. Open `annotation_sheet.csv` in a spreadsheet editor (Excel, Google Sheets, LibreOffice)
2. For each email (grouped by `email_id`), read all sentences
3. For each sentence, mark:
   - **Y** = This sentence makes the email phishing (contains malicious intent)
   - **N** = This sentence is benign (normal email content)
   - **?** = Uncertain

## What counts as a "phishing sentence"?

A sentence is phishing if it:
- Creates false urgency ("Your account will be suspended in 24 hours")
- Impersonates an authority figure ("Dear PayPal Customer")
- Requests sensitive information ("Please verify your credit card")
- Contains deceptive links or instructions ("Click here to update your account")
- Uses social manipulation ("Your friend recommended you")

A sentence is NOT phishing if it:
- Contains generic greetings without malicious intent ("Dear Customer")
- Is standard email formatting (signatures, disclaimers, footers)
- Is a neutral statement without deceptive content

## Annotation Tips

- Focus on the **intent** of the sentence, not just keywords
- Consider context: a sentence alone might seem benign but is phishing in context
- Multiple sentences can be phishing — mark all that apply
- If uncertain, mark "?" — this is better than a wrong guess
- The `model_attention` column shows which sentences the AI model focused on (for reference only — do not let it bias your judgment)
- The `rule_score` column shows keyword-based rule scores (for reference only)

## Evaluation Metrics

Your annotations will be used to compute:

1. **Precision@1**: Does the model's top-1 attended sentence match any human-annotated phishing sentence?
2. **Precision@3**: What fraction of the model's top-3 sentences are human-annotated phishing?
3. **Recall@K**: What fraction of human-annotated phishing sentences appear in the model's top-K?
4. **Human Agreement**: Cohen's κ between human annotators (if multiple annotators)

## Estimated Time

- ~150 emails × ~30 seconds each ≈ **75 minutes** for one annotator
- Recommended: 2-3 annotators for inter-annotator agreement
""")

    print(f"Annotation protocol saved to: {protocol_path}")

    # Summary stats
    stratum_counts = {}
    for e in selected:
        stratum_counts[e['stratum']] = stratum_counts.get(e['stratum'], 0) + 1

    print(f"\n=== Annotation Dataset Summary ===")
    print(f"Total emails: {len(selected)}")
    for name, count in sorted(stratum_counts.items()):
        print(f"  {name}: {count}")

    spearman_selected = [e['spearman'] for e in selected if e['spearman'] is not None]
    if spearman_selected:
        print(f"\nSpearman (selected): mean={np.mean(spearman_selected):.3f}, "
              f"median={np.median(spearman_selected):.3f}, "
              f"range=[{np.min(spearman_selected):.3f}, {np.max(spearman_selected):.3f}]")

    sent_lengths = [e['n_sentences'] for e in selected]
    print(f"Sentences/email: mean={np.mean(sent_lengths):.1f}, "
          f"median={np.median(sent_lengths):.0f}, "
          f"range=[{np.min(sent_lengths)}, {np.max(sent_lengths)}]")


if __name__ == '__main__':
    main()
