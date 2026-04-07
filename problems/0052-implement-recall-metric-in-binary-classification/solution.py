import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    total_true = np.sum(y_true)
    pred_true = np.where((y_pred + y_true) == 2, 1, 0)
    num_pred_true = np.sum(pred_true)
    return num_pred_true / total_true 