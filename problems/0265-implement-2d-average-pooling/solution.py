def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	# Your code here
	import numpy as np
	n = len(input_matrix)
	new_size = n // pool_size
	input_matrix = np.array(input_matrix)
	result = [[0 for i in range(new_size)] for j in range(new_size)]
	for i in range(0, n, pool_size):
		for j in range(0, n, pool_size):
			mean = np.mean(input_matrix[i:i+pool_size, j:j+pool_size])
			row, col = i // pool_size, j // pool_size
			result[row][col] = float(mean)
	return result