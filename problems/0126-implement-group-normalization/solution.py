import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    b, c, h, w = X.shape
    channel_per_group = int(c // num_groups)

    X_reshaped = X.reshape(b, num_groups, channel_per_group, h, w)
    gamma_reshaped = gamma.reshape(1, num_groups, channel_per_group, 1, 1)
    beta_reshaped = beta.reshape(1, num_groups, channel_per_group, 1, 1)

    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
    standardized_X = (X_reshaped - mean)/np.sqrt(var + epsilon)

    normalized_X_reshaped = standardized_X * gamma_reshaped + beta_reshaped

    final_X = normalized_X_reshaped.reshape(b, c, h, w)
    return final_X

    