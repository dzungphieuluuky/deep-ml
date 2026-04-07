import numpy as np

def get_precision_recall(y_true, y_pred):
	true_pos = np.sum(np.where((y_true + y_pred) == 2, 1, 0))
	false_pos = np.sum(np.where((y_pred - y_true) == 1, 1, 0))
	total_pos = np.sum(y_true)
	precision = true_pos / (true_pos + false_pos)
	recall = true_pos / total_pos
	return precision, recall

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	precision, recall = get_precision_recall(y_true, y_pred)
	f = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
	return round(f, 3)
