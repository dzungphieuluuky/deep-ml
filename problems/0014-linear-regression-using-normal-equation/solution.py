import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	# (X_T @ X) @ A = X_T @ y
	X = np.array(X)
	y = np.array(y)
	X_T = np.transpose(X)
	right_side = X_T @ y
	left_side = X_T @ X
	A = right_side @ np.linalg.inv(left_side)
	return A.tolist()