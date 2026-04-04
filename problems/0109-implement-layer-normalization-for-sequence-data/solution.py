import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	mean = np.mean(X, axis=-1, keepdims=True)
	std = np.std(X, axis=-1, keepdims=True)
	
	normalized_X = gamma*(X - mean)/std + beta
	return normalized_X