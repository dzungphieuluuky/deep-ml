import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	n = len(X)
	result = []
	for i in range(0, n, batch_size):
		this_X = []
		this_y = []
		for j in range(i, i + batch_size):
			if j >= n:
				break
			this_X.append(X[j])
			if y is not None:
				this_y.append(y[j])
		result.append([this_X, this_y])
	return result
