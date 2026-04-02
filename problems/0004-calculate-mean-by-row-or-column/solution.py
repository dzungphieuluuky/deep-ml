import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	means = []
	num_rows = len(matrix)
	num_columns = len(matrix[0])
	if mode == "row":
		means = np.mean(matrix, axis=1)
	elif mode == "column":
		means = np.mean(matrix, axis=0)
	return means