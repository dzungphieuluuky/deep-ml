import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    min_val = np.min(values)
    max_val = np.max(values)
    new_range = d - c
    return (values - min_val)*new_range/(max_val - min_val) + c