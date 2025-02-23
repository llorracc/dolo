import numpy
from numpy import array, zeros


def calibration_to_vector(symbols, calibration_dict):

    from dolang.triangular_solver import solve_triangular_system  # Resolve var dependencies (snt3p5)
    from numpy import nan

    sol = solve_triangular_system(calibration_dict)   # Solve for all vars in order (snt3p5)

    calibration = dict()
    for group in symbols:
        # t = numpy.array([sol.get(s, nan) for s in symbols[group]], dtype=float)
        t = numpy.array([sol.get(s, nan) for s in symbols[group]], dtype=float)
        calibration[group] = t

    return calibration


def calibration_to_dict(symbols, calib):

    if not isinstance(symbols, dict):
        symbols = symbols.symbols

    d = dict()
    for group, values in calib.items():
        if group == "covariances":                    # Skip shock correlations (snt3p5)
            continue
        syms = symbols[group]
        for i, s in enumerate(syms):
            d[s] = values[i]                          # Map each symbol to its value (snt3p5)

    return d


from dolo.compiler.misc import calibration_to_dict

import copy

equivalent_symbols = dict(actions="controls")         # Legacy name mapping (snt3p5)


class LoosyDict(dict):
    def __init__(self, **kwargs):

        kwargs = kwargs.copy()
        if "equivalences" in kwargs.keys():
            self.__equivalences__ = kwargs.pop("equivalences")  # Store name mappings (snt3p5)
        else:
            self.__equivalences__ = dict()
        super().__init__(**kwargs)

    def __getitem__(self, p):

        if p in self.__equivalences__.keys():
            k = self.__equivalences__[p]              # Map legacy to current names (snt3p5)
        else:
            k = p
        return super().__getitem__(k)


class CalibrationDict:

    # cb = CalibrationDict(symbols, calib)
    # calib['states'] -> array([ 1.        ,  9.35497829])
    # calib['states','controls'] - > [array([ 1.        ,  9.35497829]), array([ 0.23387446,  0.33      ])]
    # calib['z'] - > 1.0
    # calib['z','y'] -> [1.0, 0.99505814380953039]

    def __init__(self, symbols, calib, equivalences=equivalent_symbols):
        calib = copy.deepcopy(calib)
        for v in calib.values():
            v.setflags(write=False)                   # Make arrays immutable (snt3p5)
        self.symbols = symbols
        self.flat = calibration_to_dict(symbols, calib)  # Single-level dict (snt3p5)
        self.grouped = LoosyDict(                     # Group vars by type with aliases (snt3p5)
            **{k: v for (k, v) in calib.items()}, equivalences=equivalences
        )

    def __getitem__(self, p):
        if isinstance(p, tuple):
            return [self[e] for e in p]               # Handle multi-var lookup (snt3p5)
        if p in self.symbols.keys() or (p in self.grouped.__equivalences__.keys()):
            return self.grouped[p]                    # Return group array if group name (snt3p5)
        else:
            return self.flat[p]                       # Return single value if var name (snt3p5)


def allocating_function(inplace_function, size_output):
    def new_function(*args, **kwargs):
        val = numpy.zeros(size_output)
        nargs = args + (val,)                        # Add output array as last arg (snt3p5)
        inplace_function(*nargs)
        if "diff" in kwargs:
            return numdiff(new_function, args)       # Compute numerical derivs (snt3p5)
        return val

    return new_function


def numdiff(fun, args):
    """Vectorized numerical differentiation"""

    # vectorized version

    epsilon = 1e-8
    args = list(args)
    v0 = fun(*args)                                  # Evaluate at base point (snt3p5)
    N = v0.shape[0]
    l_v = len(v0)
    dvs = []
    for i, a in enumerate(args):
        l_a = (a).shape[1]
        dv = numpy.zeros((N, l_v, l_a))              # Store partial derivs for arg i (snt3p5)
        nargs = list(args)  # .copy()
        for j in range(l_a):
            xx = args[i].copy()
            xx[:, j] += epsilon                       # Perturb jth component (snt3p5)
            nargs[i] = xx
            dv[:, :, j] = (fun(*nargs) - v0) / epsilon  # Forward difference (snt3p5)
        dvs.append(dv)
    return [v0] + dvs                                # Return [f, df/dx1, df/dx2, ...] (snt3p5)
