def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	import numpy as np
	B = np.array(B)
	C = np.array(C)
	inv_C = np.linalg.inv(C)
	P = inv_C @ B
	return P