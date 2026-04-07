import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	x = np.array(x)
	n = x.shape[0]
	max_val = np.max(x)
	num_choice = max_val + 1
	if n_col is not None:
		num_choice = n_col
	onehot = np.zeros((n, num_choice))
	for i in range(n):
		onehot[i][x[i]] = 1
	return onehot
