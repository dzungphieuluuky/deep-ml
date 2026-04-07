import numpy as np
def precision(y_true, y_pred):
	# Your code here
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	true_pos = np.sum(np.where((y_true + y_pred) == 2, 1, 0))
	false_pos = np.sum(np.where((y_pred - y_true) == 1, 1, 0))
	return true_pos/(true_pos + false_pos)
