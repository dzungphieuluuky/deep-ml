import numpy as np

def derivative(coeffs: list):
    n = len(coeffs)
    new_coeffs = [0 for i in range(n - 1)]
    for i in range(n - 1):
        new_coeffs[i] = coeffs[i + 1] * (i + 1)
    return new_coeffs

def multiply_polynomial(x: list, y: list):
    nx = len(x)
    ny = len(y)
    nz = (nx - 1) + (ny - 1) + 1
    z = [0 for i in range(nz)]
    for i in range(nx):
        for j in range(ny):
            z[i + j] += (x[i] * y[j])
    return z

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    # Your code here
    z = multiply_polynomial(f_coeffs, g_coeffs)
    dev_z = derivative(z)
    if not dev_z:
        dev_z = [0.0]
    return dev_z
