
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	inter = np.sum(np.where((y_true + y_pred) == 2, 1, 0))
	union = np.sum(np.where((y_true + y_pred) >= 1, 1, 0))
	if y_true is None:
		return 0
	result = inter / union
	return round(result, 3)
