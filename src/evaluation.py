"""
Evaluation & Metrics Module
Computes classification metrics, confusion matrices, and cost-weighted financial loss.
"""

import numpy as np

def calculate_metrics(y_true, y_pred, c_fp=20.0, c_fn=500.0):
    """
    Computes standard classification metrics and financial loss.
    
    Args:
        y_true (np.ndarray): Binary ground truth labels (0 or 1).
        y_pred (np.ndarray): Binary predicted labels (0 or 1).
        c_fp (float): Cost per False Positive ($).
        c_fn (float): Cost per False Negative ($).
        
    Returns:
        dict: Evaluation metrics and financial cost.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    financial_loss = fp * c_fp + fn * c_fn
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
        'financial_loss': float(financial_loss)
    }

def print_metrics_summary(model_name: str, metrics: dict):
    """Prints formatted metrics summary."""
    cm = metrics['confusion_matrix']
    print(f"=== {model_name.upper()} EVALUATION ===")
    print(f"  Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print(f"  Confusion: TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}")
    print(f"  Financial Loss: ${metrics['financial_loss']:,.2f}\n")
