from scipy.stats import spearmanr, pearsonr
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from .constants import TaskType

def find_best_threshold(y_true, scores, n_thresholds=100):
    y = np.array(y_true, dtype=float)
    s = np.array(scores, dtype=float)
    mask_valid = (~np.isnan(y)) & np.isfinite(y) & np.isfinite(s)
    if not mask_valid.any():
        return 0.5, 0.0, 0.0

    y = y[mask_valid]
    s = s[mask_valid]
    uniq = set(np.unique(y))
    if uniq == {-1.0, 1.0}:
        y = (y == 1.0).astype(int)
    else:
        y = y.astype(int)

    thr_min, thr_max = np.min(s), np.max(s)
    if thr_min == thr_max:
        return float(thr_min), 0.0, accuracy_score(y, (s >= thr_min).astype(int))

    thresholds = np.linspace(0, 1, n_thresholds) 
    best_f1, best_acc, best_thr = 0.0, 0.0, thresholds[0]
    for thr in thresholds:
        y_pred = (s >= thr).astype(int)
        try:
            f1 = f1_score(y, y_pred)
        except ValueError:
            f1 = 0.0
        acc = accuracy_score(y, y_pred)
        if f1 > best_f1:
            best_f1, best_acc, best_thr = f1, acc, thr
    return float(best_thr), float(best_f1), float(best_acc)

def compute_roc_auc(y_true, scores, pos_label=1):
    y = np.array(y_true, dtype=float)
    s = np.array(scores, dtype=float)
    mask_valid = (~np.isnan(y)) & np.isfinite(y) & np.isfinite(s)
    if not mask_valid.any():
        return 0.0
    y = y[mask_valid]
    s = s[mask_valid]
    uniq = set(np.unique(y))
    if uniq == {-1.0, 1.0}:
        y = (y == float(pos_label)).astype(int)
    else:
        y = y.astype(int)
    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        auc = np.nan
    return float(auc)

def compute_metrics(eval_pred, classifier_configs: dict) -> dict:
    predictions, label_ids = eval_pred
    metrics = {}
    cls_f1_list, cls_auc_list = [], []
    reg_mse_list, reg_spear_list, reg_pear_list = [], [], []

    for key, labels in label_ids.items():
        obj = classifier_configs.get(key, {}).get("objective", "missing")
        try:
            task_type = TaskType.from_str(obj)
        except ValueError:
            task_type = None

        if task_type == TaskType.PI:
            preds = np.array(predictions[key], dtype=float)
            labels = np.array(labels, dtype=float)
            mask = (~np.isnan(labels)) & np.isfinite(labels) & np.isfinite(preds)
            if not mask.any():
                continue
            y_true = labels[mask]
            y_pred_raw = preds[mask]
            thr, f1, acc = find_best_threshold(y_true, y_pred_raw)
            auc = compute_roc_auc(y_true, y_pred_raw)
            metrics[f"{key}_best_thr"] = thr
            metrics[f"{key}_best_f1"]  = f1
            metrics[f"{key}_best_acc"] = acc
            metrics[f"{key}_auc"]      = auc
            metrics[f"{key}_mse"]      = float(mean_squared_error(y_true, y_pred_raw))
            cls_f1_list.append(f1)
            cls_auc_list.append(auc)

        elif task_type == TaskType.STS:
            preds = np.array(predictions[key], dtype=float)
            labels = np.array(labels, dtype=float)
            mask = (~np.isnan(labels)) & np.isfinite(labels) & np.isfinite(preds)
            if not mask.any():
                continue
            y_true = labels[mask]
            y_pred_raw = preds[mask]
            y_pred_flat = y_pred_raw.flatten()
            mse = mean_squared_error(y_true, y_pred_flat)
            try:
                spear, _ = spearmanr(y_pred_flat, y_true)
            except:
                spear = 0.0
            try:
                pear, _ = pearsonr(y_pred_flat, y_true)
            except:
                pear = 0.0
            metrics[f"{key}_mse"]      = float(mse)
            metrics[f"{key}_spearman"] = float(np.nan_to_num(spear))
            metrics[f"{key}_pearson"]  = float(np.nan_to_num(pear))
            reg_mse_list.append(mse)
            reg_spear_list.append(spear)
            reg_pear_list.append(pear)

        elif task_type == TaskType.TRIPLET:
            preds = np.array(predictions[f"{key}_anchor_prob"], dtype=float)
            labels = np.array(labels, dtype=float)
            mask = (~np.isnan(labels)) & np.isfinite(labels)
            if not mask.any():
                continue
            y_true = labels[mask]
            y_pred_probs = np.array(predictions[f"{key}_anchor_prob"], dtype=float)[mask]
            y_true_int   = y_true.astype(int)
            y_pred_label = np.argmax(y_pred_probs, axis=1)
            acc = accuracy_score(y_true_int, y_pred_label)
            f1  = f1_score(y_true_int, y_pred_label, average="macro")
            average_positive_cosine_similarity = np.mean(predictions[f"{key}_pos_similarity"][mask])
            average_negative_cosine_similarity = np.mean(predictions[f"{key}_neg_similarity"][mask])
            triplet_accuracy = np.mean(
                predictions[f"{key}_pos_similarity"][mask] > predictions[f"{key}_neg_similarity"][mask]
            )
            metrics[f"{key}_triplet_accuracy"] = float(triplet_accuracy)
            metrics[f"{key}_average_positive_cosine_similarity"] = float(average_positive_cosine_similarity)
            metrics[f"{key}_average_negative_cosine_similarity"] = float(average_negative_cosine_similarity)
            metrics[f"{key}_accuracy"] = float(acc)
            metrics[f"{key}_macro_f1"] = float(f1)
            cls_f1_list.append(f1)

    if cls_f1_list:
        metrics["average_f1"]  = float(np.mean(cls_f1_list))
        metrics["average_auc"] = float(np.mean(cls_auc_list)) if cls_auc_list else 0.0
    if reg_mse_list:
        metrics["average_mse"]      = float(np.mean(reg_mse_list))
        metrics["average_spearman"] = float(np.mean(reg_spear_list))
        metrics["average_pearson"]  = float(np.mean(reg_pear_list))
    return metrics
