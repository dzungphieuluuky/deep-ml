import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	height, width, channel = x.shape
	return np.mean(x, axis=(0, 1), keepdims=False)