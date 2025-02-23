import ast
from ast import BinOp, Compare, Sub
import sympy                # Symbolic mathematics lib for eqn manipulation (snt3p5)
import numpy

from dolo.compiler.function_compiler_sympy import (
    ast_to_sympy,
    compile_higher_order_function,
)

from dolang.symbolic import stringify, stringify_symbol


def timeshift(expr, variables, date):
    from sympy import Symbol
    from dolang import stringify

    d = {
        Symbol(stringify_symbol((v, 0))): Symbol(stringify_symbol((v, date)))  # Map vars from t=0 to target date (snt3p5)
        for v in variables
    }
    return expr.subs(d)  # Apply time shift substitution to expr (snt3p5)


def parse_equation(eq_string, vars, substract_lhs=True, to_sympy=False):

    eq = eq_string.split("|")[0]  # ignore complentarity constraints

    if "==" not in eq:
        eq = eq.replace("=", "==")  # Normalize equation syntax to use == (snt3p5)

    expr = ast.parse(eq).body[0].value
    expr_std = stringify(expr, variables=vars)  # Convert to dolang's canonical form (snt3p5)

    if isinstance(expr_std, Compare):
        lhs = expr_std.left
        rhs = expr_std.comparators[0]
        if substract_lhs:
            expr_std = BinOp(left=rhs, right=lhs, op=Sub())  # Convert a=b to b-a for residual form (snt3p5)
        else:
            if to_sympy:
                return [ast_to_sympy(lhs), ast_to_sympy(rhs)]
            return [lhs, rhs]

    if to_sympy:
        return ast_to_sympy(expr_std)
    else:
        return expr_std


def model_to_fg(model, order=2):
    # compile f, g function at higher order

    all_variables = sum(
        [model.symbols[e] for e in model.symbols if e != "parameters"], []
    )
    all_dvariables = (
        [(d, 0) for d in all_variables]                # Current period vars t (snt3p5)
        + [(d, 1) for d in all_variables]              # Next period vars t+1 (snt3p5)
        + [(d, -1) for d in all_variables]             # Previous period vars t-1 (snt3p5)
    )
    psyms = [(e, 0) for e in model.symbols["parameters"]]

    definitions = model.definitions

    d = dict()

    for k in definitions:
        v = parse_equation(definitions[k], all_variables, to_sympy=True)
        kk = stringify_symbol((k, 0))                  # Current period symbol (snt3p5)
        kk_m1 = stringify_symbol((k, -1))             # Previous period symbol (snt3p5)
        kk_1 = stringify_symbol((k, 1))               # Next period symbol (snt3p5)
        d[sympy.Symbol(kk)] = v
        d[sympy.Symbol(kk_m1)] = timeshift(v, all_variables, -1)  # Create t-1 version (snt3p5)
        d[sympy.Symbol(kk_1)] = timeshift(v, all_variables, 1)    # Create t+1 version (snt3p5)

    f_eqs = model.equations["arbitrage"]              # Optimality conditions (snt3p5)
    f_eqs = [parse_equation(eq, all_variables, to_sympy=True) for eq in f_eqs]
    f_eqs = [eq.subs(d) for eq in f_eqs]             # Substitute definitions into eqns (snt3p5)

    g_eqs = model.equations["transition"]             # State transition eqns (snt3p5)
    g_eqs = [
        parse_equation(eq, all_variables, to_sympy=True, substract_lhs=False)
        for eq in g_eqs
    ]
    # solve_recursively
    dd = dict()
    for eq in g_eqs:
        dd[eq[0]] = eq[1].subs(dd).subs(d)           # Build recursive substitution chain (snt3p5)
    g_eqs = dd.values()

    f_syms = (
        [(e, 0) for e in model.symbols["states"]]     # Current states (snt3p5)
        + [(e, 0) for e in model.symbols["controls"]] # Current controls (snt3p5)
        + [(e, 1) for e in model.symbols["states"]]   # Next period states (snt3p5)
        + [(e, 1) for e in model.symbols["controls"]] # Next period controls (snt3p5)
    )

    g_syms = (
        [(e, -1) for e in model.symbols["states"]]    # Previous states (snt3p5)
        + [(e, -1) for e in model.symbols["controls"]]# Previous controls (snt3p5)
        + [(e, 0) for e in model.symbols["exogenous"]]# Current shocks (snt3p5)
    )

    params = model.symbols["parameters"]

    f = compile_higher_order_function(                # Create optimality fn (snt3p5)
        f_eqs,
        f_syms,
        params,
        order=order,
        funname="f",
        return_code=False,
        compile=False,
    )

    g = compile_higher_order_function(                # Create transition fn (snt3p5)
        g_eqs,
        g_syms,
        params,
        order=order,
        funname="g",
        return_code=False,
        compile=False,
    )
    # cache result
    model.__higher_order_functions__ = dict(f=f, g=g)
    model.__highest_order__ = order

    return [f, g]


def get_model_derivatives(model, order, calibration=None):

    if calibration is None:
        calibration = model.calibration

    [f_fun, g_fun] = model_to_fg(model, order=order)  # Get optimality and transition fns (snt3p5)

    parms = model.calibration["parameters"]
    states_ss = calibration["states"]                 # Steady state values (snt3p5)
    controls_ss = calibration["controls"]
    shocks_ss = calibration["exogenous"]

    f_args_ss = numpy.concatenate([states_ss, controls_ss, states_ss, controls_ss])  # Pack steady state args (snt3p5)
    g_args_ss = numpy.concatenate([states_ss, controls_ss, shocks_ss])

    f = f_fun(f_args_ss, parms, order=order)         # Evaluate derivs at steady state (snt3p5)
    g = g_fun(g_args_ss, parms, order=order)

    return [f, g]
