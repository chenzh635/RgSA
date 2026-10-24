"""ERASER-style faithfulness metrics: comprehensiveness & sufficiency.

These metrics break the circularity of the rule-based Spearman metric:
instead of measuring agreement with the training signal (rule scores),
they measure whether the sentences a model selects actually drive its
own classification decision (Jain & Wallace 2019; DeYoung et al. 2020).

Protocol (for models with selection weights w over sentences and an
email-level logit computed from a weighted sentence pool z = sum w_i h_i):

  top-k     : the k valid sentences with largest selection weight
  Comp@k    : logit_full - logit(top-k removed, remaining weights
              renormalized to sum 1)                      [higher = better]
  Suff@k    : logit_full - logit(only top-k kept, weights renormalized
              within the kept set)                        [lower = better]
  Rand@k    : Comp@k with k uniformly random sentences instead of top-k
              (matched-size control; advantage = Comp - Rand)

Design choices:
- All primary values are computed in LOGIT space because classification
  probabilities saturate near 1 on this dataset; probability-space
  values are recorded alongside.
- Primary reporting scope: correctly classified phishing emails
  (p_full > 0.5), following ERASER convention; all-phishing values
  are recorded as well.
- Emails with < 2 valid sentences are excluded (degenerate ranking;
  ~25% of phishing emails). For Comp@k, emails with n_valid <= k are
  excluded; cases where removal leaves zero residual weight ("all mass
  in top-k") are counted separately rather than silently dropped.
- Random control uses a fixed RNG seed for reproducibility.
"""

import numpy as np
import torch

KS_DEFAULT = (1, 3, 5)
N_RANDOM = 5
RNG_SEED = 0
EPS = 1e-8


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _topk_indices(w, n_valid, k):
    """Indices of the k highest-weight sentences among the first n_valid."""
    order = np.argsort(-w[:n_valid], kind='stable')
    return order[:k]


def _pooled_logit(h, weights, classify_fn):
    """Pool sentence embeddings with `weights` and return classifier logit."""
    z = (h * weights[:, None]).sum(axis=0)
    return classify_fn(z)


def make_classifier_fn(module, device):
    """Wrap an MLP (e.g. model.classifier / model.predictor) as z -> logit."""
    module.eval()

    @torch.no_grad()
    def fn(z):
        t = torch.from_numpy(np.asarray(z, dtype=np.float32)).unsqueeze(0).to(device)
        return float(module(t).squeeze(-1).item())

    return fn


def faithfulness_pooling_email(h, w, n_valid, logit_full, classify_fn,
                               ks, rng, n_random=N_RANDOM):
    """Per-email faithfulness values for a pooling-style model.

    Args:
        h: (S, D) sentence embeddings (numpy)
        w: (S,) selection weights, renormalized to sum 1 over valid entries
        n_valid: number of valid sentences
        logit_full: email logit with all sentences
        classify_fn: pooled embedding -> logit
    Returns:
        dict k -> {'comp_logit', 'comp_prob', 'suff_drop_logit',
                   'suff_keep_prob', 'rand_comp_logit', 'comp_concentrated'}
        (entries None where undefined)
    """
    prob_full = float(_sigmoid(logit_full))
    scale = max(abs(logit_full), EPS)
    out = {}
    for k in ks:
        entry = {
            'comp_logit': None, 'comp_prob': None, 'comp_rel': None,
            'suff_drop_logit': None, 'suff_keep_prob': None,
            'rand_comp_logit': None, 'comp_concentrated': False,
        }
        topk = _topk_indices(w, n_valid, k)

        # Comprehensiveness: remove top-k (needs at least one sentence left)
        if n_valid > k:
            keep = np.ones(n_valid, dtype=bool)
            keep[topk] = False
            rest_sum = w[:n_valid][keep].sum()
            if rest_sum > EPS:
                w_comp = np.zeros(len(w))
                w_comp[:n_valid][keep] = w[:n_valid][keep] / rest_sum
                logit_comp = _pooled_logit(h, w_comp, classify_fn)
                entry['comp_logit'] = float(logit_full - logit_comp)
                entry['comp_prob'] = float(prob_full - _sigmoid(logit_comp))
                entry['comp_rel'] = float((logit_full - logit_comp) / scale)
            else:
                entry['comp_concentrated'] = True

        # Sufficiency: keep only top-k
        keep = np.zeros(len(w), dtype=bool)
        keep[topk] = True
        top_sum = w[:n_valid][keep[:n_valid]].sum()
        if top_sum > EPS:
            w_suff = np.zeros(len(w))
            w_suff[:n_valid][keep[:n_valid]] = w[:n_valid][keep[:n_valid]] / top_sum
            logit_suff = _pooled_logit(h, w_suff, classify_fn)
            entry['suff_drop_logit'] = float(logit_full - logit_suff)
            entry['suff_keep_prob'] = float(_sigmoid(logit_suff))

        # Random-k control
        if n_valid > k:
            vals = []
            for _ in range(n_random):
                idx = rng.choice(n_valid, size=k, replace=False)
                keep_r = np.ones(n_valid, dtype=bool)
                keep_r[idx] = False
                rest_sum = w[:n_valid][keep_r].sum()
                if rest_sum > EPS:
                    w_r = np.zeros(len(w))
                    w_r[:n_valid][keep_r] = w[:n_valid][keep_r] / rest_sum
                    vals.append(logit_full - _pooled_logit(h, w_r, classify_fn))
            if vals:
                entry['rand_comp_logit'] = float(np.mean(vals))

        out[k] = entry
    return out


class FaithfulnessAggregator:
    """Accumulates per-email values, split by scope (all / correct)."""

    def __init__(self, ks):
        self.ks = list(ks)
        self.n_phishing = 0
        self.n_correct = 0
        self.n_short_excluded = 0          # n_valid < 2
        self.n_no_selection = 0            # degenerate weights (e.g. rules silent)
        self.counts = {k: {'n_comp': 0, 'n_comp_concentrated': 0, 'n_suff': 0,
                           'n_rand': 0} for k in self.ks}
        self.vals = {k: {m: [] for m in
                         ('comp_logit', 'comp_prob', 'comp_rel', 'suff_drop_logit',
                          'suff_keep_prob', 'rand_comp_logit')}
                     for k in self.ks}
        self.logits_full = []
        self.comp_logit_correct = {k: [] for k in self.ks}
        self.comp_prob_correct = {k: [] for k in self.ks}
        self.comp_rel_correct = {k: [] for k in self.ks}
        self.suff_drop_logit_correct = {k: [] for k in self.ks}
        self.suff_keep_prob_correct = {k: [] for k in self.ks}
        self.rand_comp_logit_correct = {k: [] for k in self.ks}

    def add(self, logit_full, per_k):
        """per_k: output of faithfulness_pooling_email. Scope decided here."""
        correct = logit_full > 0
        self.logits_full.append(float(logit_full))
        for k in self.ks:
            e = per_k.get(k, {})
            for m in ('comp_logit', 'comp_prob', 'comp_rel', 'suff_drop_logit',
                      'suff_keep_prob', 'rand_comp_logit'):
                v = e.get(m)
                if v is not None:
                    self.vals[k][m].append(v)
                    if correct:
                        getattr(self, f'{m}_correct')[k].append(v)
            if e.get('comp_logit') is not None:
                self.counts[k]['n_comp'] += 1
            if e.get('comp_concentrated'):
                self.counts[k]['n_comp_concentrated'] += 1
            if e.get('suff_drop_logit') is not None:
                self.counts[k]['n_suff'] += 1
            if e.get('rand_comp_logit') is not None:
                self.counts[k]['n_rand'] += 1

    def summarize(self):
        def mean(lst):
            return float(np.mean(lst)) if lst else None

        out = {
            'n_phishing': self.n_phishing,
            'n_correct': self.n_correct,
            'n_short_excluded': self.n_short_excluded,
            'n_no_selection': self.n_no_selection,
            'mean_logit_full': mean(self.logits_full),
            'per_k': {},
        }
        for k in self.ks:
            out['per_k'][str(k)] = {
                # all-phishing scope
                'comp_logit_all': mean(self.vals[k]['comp_logit']),
                'comp_rel_all': mean(self.vals[k]['comp_rel']),
                'rand_comp_logit_all': mean(self.vals[k]['rand_comp_logit']),
                'suff_drop_logit_all': mean(self.vals[k]['suff_drop_logit']),
                'suff_keep_prob_all': mean(self.vals[k]['suff_keep_prob']),
                'comp_advantage_all': (
                    mean(self.vals[k]['comp_logit']) - mean(self.vals[k]['rand_comp_logit'])
                    if self.vals[k]['comp_logit'] and self.vals[k]['rand_comp_logit']
                    else None),
                # correctly-classified scope (primary, ERASER convention)
                'comp_logit_correct': mean(self.comp_logit_correct[k]),
                'comp_prob_correct': mean(self.comp_prob_correct[k]),
                'comp_rel_correct': mean(self.comp_rel_correct[k]),
                'suff_drop_logit_correct': mean(self.suff_drop_logit_correct[k]),
                'suff_keep_prob_correct': mean(self.suff_keep_prob_correct[k]),
                'rand_comp_logit_correct': mean(self.rand_comp_logit_correct[k]),
                'comp_advantage_correct': (
                    mean(self.comp_logit_correct[k]) - mean(self.rand_comp_logit_correct[k])
                    if self.comp_logit_correct[k] and self.rand_comp_logit_correct[k]
                    else None),
                **self.counts[k],
            }
        return out


def evaluate_faithfulness_pooling(model, kind, test_loader, device,
                                  ks=KS_DEFAULT, n_random=N_RANDOM,
                                  classify_attr='classifier'):
    """Faithfulness metrics for any model that pools selection-weighted
    sentence embeddings through an MLP head.

    kind: one of 'cnn_sparse', 'cnn_abmil', 'bert', 'bert_sparse', 'ai2tale'
    """
    model.eval()
    classify_fn = make_classifier_fn(getattr(model, classify_attr), device)
    rng = np.random.default_rng(RNG_SEED)
    agg = FaithfulnessAggregator(ks)

    with torch.no_grad():
        for batch in test_loader:
            sent_lengths = batch['sent_lengths'].to(device)
            if kind == 'cnn_sparse':
                output, attn_w, _, h = model(
                    batch['tokens'].to(device), sent_lengths, temperature=0.1)
            elif kind == 'cnn_abmil':
                output, attn_w, h = model(batch['tokens'].to(device), sent_lengths)
            elif kind == 'bert':
                output, attn_w, h = model(batch['raw_sentences'], sent_lengths)
            elif kind == 'bert_sparse':
                output, attn_w, _, h = model(
                    batch['raw_sentences'], sent_lengths, temperature=0.1)
            elif kind == 'ai2tale':
                output, attn_w, h = model(batch['tokens'].to(device), sent_lengths)
            else:
                raise ValueError(f'unknown kind {kind}')

            logits = output.cpu().numpy().astype(np.float64)
            labels = batch['label'].numpy()
            sl = batch['sent_lengths'].numpy()
            w_all = attn_w.cpu().numpy().astype(np.float64)
            h_all = h.cpu().numpy().astype(np.float32)

            raw_sentences = batch['raw_sentences']
            for b in range(len(labels)):
                if labels[b] != 1:
                    continue
                agg.n_phishing += 1
                if kind in ('bert', 'bert_sparse'):
                    # BERT encodes decoded sentences only; sentences whose
                    # tokens are all out-of-vocab decode to nothing and are
                    # absent from raw_sentences (sent_lengths would
                    # overcount them).
                    n_valid = min(len(raw_sentences[b]), w_all.shape[1])
                else:
                    n_valid = min(int((sl[b] > 0).sum()), w_all.shape[1])
                if logits[b] > 0:
                    agg.n_correct += 1
                if n_valid < 2:
                    agg.n_short_excluded += 1
                    continue
                w = w_all[b].copy()
                w[n_valid:] = 0.0
                s = w[:n_valid].sum()
                if s <= EPS:
                    agg.n_no_selection += 1
                    continue
                w[:n_valid] /= s
                per_k = faithfulness_pooling_email(
                    h_all[b], w, n_valid, logits[b], classify_fn, ks, rng, n_random)
                agg.add(logits[b], per_k)

    return agg.summarize()


def evaluate_faithfulness_snorkel(model, test_loader, device,
                                  ks=KS_DEFAULT, n_random=N_RANDOM):
    """Faithfulness for the Snorkel sentence classifier.

    Selection weights: softmax over per-sentence logits (as in the
    paper's evaluation). Decision rule: max over sentence logits.
    Comp@k = logit_max - max(remaining); Suff@k = logit_max - max(top-k).
    """
    model.eval()
    rng = np.random.default_rng(RNG_SEED)
    agg = FaithfulnessAggregator(ks)

    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label'].numpy()
            tokens = batch['tokens'].to(device)
            raw_sentences = batch['raw_sentences']

            for b in range(len(labels)):
                if labels[b] != 1:
                    continue
                agg.n_phishing += 1
                n_valid = len(raw_sentences[b])
                if n_valid < 2:
                    agg.n_short_excluded += 1
                    continue
                sent_tokens = tokens[b, :n_valid, :]
                logits_s = model(sent_tokens).cpu().numpy().astype(np.float64)
                logit_full = float(logits_s.max())
                if logit_full > 0:
                    agg.n_correct += 1

                stable = logits_s - logits_s.max()
                w = np.exp(stable)
                w /= w.sum()

                per_k = {}
                for k in ks:
                    entry = {'comp_logit': None, 'comp_prob': None, 'comp_rel': None,
                             'suff_drop_logit': None, 'suff_keep_prob': None,
                             'rand_comp_logit': None, 'comp_concentrated': False}
                    topk = np.argsort(-w, kind='stable')[:k]
                    if n_valid > k:
                        keep = np.ones(n_valid, dtype=bool)
                        keep[topk] = False
                        logit_comp = float(logits_s[keep].max())
                        entry['comp_logit'] = logit_full - logit_comp
                        entry['comp_prob'] = float(_sigmoid(logit_full) - _sigmoid(logit_comp))
                        entry['comp_rel'] = float(
                            (logit_full - logit_comp) / max(abs(logit_full), EPS))
                        vals = []
                        for _ in range(n_random):
                            idx = rng.choice(n_valid, size=k, replace=False)
                            keep_r = np.ones(n_valid, dtype=bool)
                            keep_r[idx] = False
                            vals.append(logit_full - float(logits_s[keep_r].max()))
                        entry['rand_comp_logit'] = float(np.mean(vals))
                    entry['suff_drop_logit'] = logit_full - float(logits_s[topk].max())
                    entry['suff_keep_prob'] = float(_sigmoid(logits_s[topk].max()))
                    per_k[k] = entry
                agg.add(logit_full, per_k)

    return agg.summarize()


def evaluate_faithfulness_rules_ref(ref_model, test_loader, config, device,
                                    ks=KS_DEFAULT, n_random=N_RANDOM):
    """Rules-only selection evaluated against a learned reference classifier.

    Selection weights: normalized domain-rule scores. Decision maker:
    ref_model (a trained ABMIL). Reports how necessary/sufficient the
    rule-chosen sentences are for the learned model's decision, plus
    rule coverage (fraction of phishing emails with any rule hit).
    """
    import os

    from src.rules import batch_score, load_keywords

    keywords_path = config.get('keywords_file', 'phishingdata/kwords_stem.json')
    if not os.path.isabs(keywords_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keywords_path = os.path.join(project_root, keywords_path)
    keywords = load_keywords(keywords_path)

    ref_model.eval()
    classify_fn = make_classifier_fn(ref_model.classifier, device)
    rng = np.random.default_rng(RNG_SEED)
    agg = FaithfulnessAggregator(ks)

    with torch.no_grad():
        for batch in test_loader:
            sent_lengths = batch['sent_lengths'].to(device)
            output, _, h = ref_model(batch['tokens'].to(device), sent_lengths)

            logits = output.cpu().numpy().astype(np.float64)
            labels = batch['label'].numpy()
            sl = batch['sent_lengths'].numpy()
            h_all = h.cpu().numpy().astype(np.float32)
            rule_scores = batch_score(batch['raw_sentences'], keywords)

            for b in range(len(labels)):
                if labels[b] != 1:
                    continue
                agg.n_phishing += 1
                if logits[b] > 0:
                    agg.n_correct += 1
                n_valid = int((sl[b] > 0).sum())
                if n_valid < 2:
                    agg.n_short_excluded += 1
                    continue
                S = h_all.shape[1]
                w = np.zeros(S, dtype=np.float64)
                n_use = min(n_valid, rule_scores.shape[1], S)
                w[:n_use] = rule_scores[b, :n_use]
                s = w[:n_valid].sum()
                if s <= EPS:
                    agg.n_no_selection += 1   # rules silent on this email
                    continue
                w[:n_valid] /= s
                per_k = faithfulness_pooling_email(
                    h_all[b], w, n_valid, logits[b], classify_fn, ks, rng, n_random)
                agg.add(logits[b], per_k)

    summary = agg.summarize()
    n_eval = agg.n_phishing - agg.n_short_excluded
    summary['rule_coverage'] = (
        float(1.0 - agg.n_no_selection / n_eval) if n_eval > 0 else None)
    return summary
