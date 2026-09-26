"""F_0.5-aware threshold selection and prediction."""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import fbeta_score


def macro_f05(y_true_groups, y_pred_groups):
    """Macro F_0.5 over S1 entities."""
    f_scores = []
    for s1_id in y_true_groups:
        true_set = y_true_groups[s1_id]
        pred_set = y_pred_groups[s1_id]
        if len(pred_set) == 0 and len(true_set) == 0:
            f_scores.append(1.0)
            continue
        if len(pred_set) == 0 or len(true_set) == 0:
            f_scores.append(0.0)
            continue
        tp = len(true_set & pred_set)
        if tp == 0:
            f_scores.append(0.0)
            continue
        p = tp / len(pred_set)
        r = tp / len(true_set)
        f = (1.25 * p * r) / (0.25 * p + r)
        f_scores.append(f)
    return np.mean(f_scores)


def find_best_threshold(probs, s1_ids, cand_ids, labels, threshold_grid=None):
    """Find threshold maximizing macro F_0.5."""
    if threshold_grid is None:
        threshold_grid = np.arange(0.3, 0.85, 0.02)
    best_t, best_f = 0.5, 0
    for t in threshold_grid:
        truth = defaultdict(set)
        preds = defaultdict(set)
        for s1, cand, p, y in zip(s1_ids, cand_ids, probs, labels):
            if y == 1: truth[s1].add(cand)
            if p >= t: preds[s1].add(cand)
        f = macro_f05(truth, preds)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f