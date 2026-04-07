
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	counts = Counter()
	for row in data:
		counts[(row[0], row[1])] += 1
	true_pos = counts[(1, 1)]
	true_neg = counts[(0, 0)]
	false_pos = counts[(0, 1)]
	false_neg = counts[(1, 0)]
	return [[true_pos, false_neg], [false_pos, true_neg]]