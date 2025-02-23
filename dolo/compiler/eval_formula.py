# from ast import *
from dolang.symbolic import stringify, stringify_symbol, parse_string, list_variables  # Symbolic expr processing utils (snt3p5)
from dolang.grammar import str_expression  # Convert AST to string expr (snt3p5)

from dolo.compiler.misc import CalibrationDict
from numpy import log, exp

import xarray


def eval_formula(expr: str, dataframe=None, context=None):
    """
    expr: string
        Symbolic expression to evaluate.
        Example: `k(1)-delta*k(0)-i`
    table: (optional) pandas dataframe
        Each column is a time series, which can be indexed with dolo notations.
    context: dict or CalibrationDict
    """

    if context is None:
        dd = {}  # context dictionary
    elif isinstance(context, CalibrationDict):
        dd = context.flat.copy()                      # Flatten nested calib dict (snt3p5)
    else:
        dd = context.copy()

    # compat since normalize form for parameters doesn't match calib dict.
    for k in [*dd.keys()]:
        dd[stringify_symbol(k)] = dd[k]               # Add normalized symbol versions (snt3p5)

    expr_ast = parse_string(expr)                     # Convert string to AST (snt3p5)
    variables = list_variables(expr_ast)              # Extract time-indexed vars (snt3p5)
    nexpr = stringify(expr_ast)                       # Convert to canonical form (snt3p5)

    dd["log"] = log                                   # Add math fns to eval context (snt3p5)
    dd["exp"] = exp

    if dataframe is not None:

        import pandas as pd

        for k, t in variables:
            dd[stringify_symbol((k, t))] = dataframe[k].shift(t)  # Map time-shifted vars to data (snt3p5)
        dd["t_"] = pd.Series(dataframe.index, index=dataframe.index)  # Add time index (snt3p5)

    expr = str_expression(nexpr)                      # Convert to evaluatable string (snt3p5)

    res = eval(expr.replace("^", "**"), dd)          # Eval with Python power op (snt3p5)

    return res
