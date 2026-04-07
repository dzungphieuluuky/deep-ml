def attention_memory_flops(B: int, h: int, N: int, d: int, bytes_per_element: int = 2) -> dict:
    """
    Compute memory traffic and FLOPs for standard self-attention.

    Args:
        B: Batch size
        h: Number of attention heads
        N: Sequence length
        d: Head dimension
        bytes_per_element: Bytes per element (e.g., 2 for FP16, 4 for FP32)

    Returns:
        dict with keys:
            'qk_flops': int - FLOPs for Q @ K^T
            'softmax_flops': int - FLOPs for softmax
            'pv_flops': int - FLOPs for P @ V
            'total_flops': int - Total FLOPs
            'memory_bytes': int - Total memory traffic in bytes
            'arithmetic_intensity': float - FLOPs per byte, rounded to 2 decimal places
    """
    # Your code here
    qk_flops = 2*B*h*(N**2)*d
    softmax_flops = 5*B*h*(N**2)
    pv_flops = 2*B*h*(N**2)*d
    total_flops = qk_flops + softmax_flops + pv_flops
    elements = 4*B*h*N*(d + N)
    memory_bytes = bytes_per_element * elements
    arithmetic_intensity = total_flops / memory_bytes
    return {
        "qk_flops": round(qk_flops, 2),
        "softmax_flops": round(softmax_flops, 2),
        "pv_flops": round(pv_flops, 2),
        "total_flops": round(total_flops, 2),
        "memory_bytes": round(memory_bytes, 2),
        "arithmetic_intensity": round(arithmetic_intensity, 2)
    }