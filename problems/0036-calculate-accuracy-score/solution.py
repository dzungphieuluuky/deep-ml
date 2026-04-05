import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	if y_true.shape != y_pred.shape:
		return -1
	return np.sum(y_true == y_pred)/y_true.shape[0]