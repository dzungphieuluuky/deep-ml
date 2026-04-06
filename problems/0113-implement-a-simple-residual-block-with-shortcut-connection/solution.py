import numpy as np

def relu(x: np.array):
	return np.where(x > 0, x, 0)
def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	x_1 = w1 @ x
	x_2 = w2 @ x_1
	activation = relu(x_2)
	return relu(activation + x)