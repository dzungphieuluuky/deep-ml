import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here
    batch_size, dim = x.shape
    mid = dim // 2
    coeff = x[:, :mid]
    logits = x[:, mid:]
    sigmoid = 1 / (1 + np.exp(-logits))
    swish = logits * sigmoid
    scores = coeff * swish
    return scores