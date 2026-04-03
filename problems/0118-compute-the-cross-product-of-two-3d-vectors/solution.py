import numpy as np

def cross_product(a, b):
    # Your code here
    if len(a) != 3 or len(b) != 3:
        return -1
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return [x, y, z]