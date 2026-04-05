import numpy as np

def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    """
    Calculate the variance of a portfolio.

    Args:
        cov_matrix (list[list[float]]): Covariance matrix of asset returns.
        weights (list[float]): Portfolio weights.

    Returns:
        float: Portfolio variance.
    """
    # Your code here
    weights_array = np.array(weights)
    cov_matrix = np.array(cov_matrix)
    return np.transpose(weights_array) @ cov_matrix @ weights_array