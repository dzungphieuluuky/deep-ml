import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	data = np.array(data)
	mean = np.mean(data, axis=0, keepdims=True)
	std = np.std(data, axis=0, keepdims=True)

	min_val = np.min(data, axis=0, keepdims=True)
	max_val = np.max(data, axis=0, keepdims=True)

	standardized_data = (data - mean)/std
	normalized_data = (data - min_val)/(max_val - min_val)
	return standardized_data, normalized_data