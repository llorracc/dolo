import numpy as np


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]        # Convert inputs to numpy arrays (snt3p5)
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])           # Total size of output array (snt3p5)
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)  # Initialize output array (snt3p5)

    m = n // arrays[0].size                         # Size of each sub-cartesian product (snt3p5)
    out[:, 0] = np.repeat(arrays[0], m)            # Fill first column with repeated values (snt3p5)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])    # Recursively fill remaining columns (snt3p5)
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]  # Copy pattern for each first value (snt3p5)
    return out


def mlinspace(a, b, orders, out=None):
    """Multi-dimensional version of linspace"""      # Add docstring summary (snt3p5)

    import numpy

    sl = [numpy.linspace(a[i], b[i], orders[i]) for i in range(len(a))]  # Create 1D grids (snt3p5)

    if out is None:
        out = cartesian(sl)                         # Generate grid points from linspaces (snt3p5)
    else:
        cartesian(sl, out)                          # Fill provided array with grid points (snt3p5)

    return out


def MyJacobian(fun, eps=1e-6):
    """Compute Jacobian using central differences""" # Add docstring summary (snt3p5)
    def rfun(x):
        n = len(x)
        x0 = x.copy()
        y0 = fun(x)
        D = np.zeros((len(y0), len(x0)))           # Initialize Jacobian matrix (snt3p5)
        for i in range(n):
            delta = np.zeros(len(x))
            delta[i] = eps                          # Perturbation in i-th direction (snt3p5)
            y1 = fun(x + delta)
            y2 = fun(x - delta)
            D[:, i] = (y1 - y2) / eps / 2          # Central difference approximation (snt3p5)
        return D

    return rfun
