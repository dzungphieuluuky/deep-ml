def softsign(x: float) -> float:
	"""
	Implements the Softsign activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Softsign of the input	"""
	# Your code here
	import math
	val = x/(1 + math.sqrt(x**2))
	return round(val,4)