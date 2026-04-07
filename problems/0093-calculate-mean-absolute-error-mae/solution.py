import numpy as np

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.

    Parameters:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values

    Returns:
        float: Mean Absolute Error
    """
    # Your code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.sqrt((y_true - y_pred)**2))
    return mae