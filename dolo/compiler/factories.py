# plainformatter = get_ipython().display_formatter.formatters['text/plain']
# del plainformatter.type_printers[dict]

import yaml
import numpy as np
from typing import List

import ast
from ast import BinOp, Sub

from typing import Dict

import dolang                                         # Domain-specific lang for econ models (snt3p5)
from dolang.grammar import str_expression
from dolang.symbolic import parse_string, time_shift, Sanitizer  # Symbolic manipulation utils (snt3p5)
from dolang.symbolic import time_shift
from dolang.symbolic import Sanitizer
from dolang.factory import FlatFunctionFactory        # Creates optimized numeric fns (snt3p5)


def get_name(e):
    return ast.parse(e).body[0].value.func.id         # Extract fn name from expr (snt3p5)


def reorder_preamble(pr):

    from dolang.triangular_solver import triangular_solver, get_incidence

    inc = get_incidence(pr)                          # Build dependency graph (snt3p5)
    order = triangular_solver(inc)                    # Sort vars by dependencies (snt3p5)
    d = dict()
    prl = [*pr.items()]
    for o in order:
        k, v = prl[o]
        d[k] = v
    return d                                         # Return ordered definitions (snt3p5)


def shift_spec(specs, tshift):
    ss = dict()
    if "target" in specs:
        e = specs["target"]
        ss["target"] = [e[0], e[1] + tshift, e[2]]   # Adjust target time period (snt3p5)
    ss["eqs"] = [
        ([e[0], e[1] + tshift, e[2]] if e[0] != "parameters" else e)  # Shift all non-param vars (snt3p5)
        for e in specs["eqs"]
    ]
    return ss


def get_factory(model, eq_type: str, tshift: int = 0):

    from dolo.compiler.model import decode_complementarity

    from dolo.compiler.recipes import recipes
    from dolang.symbolic import stringify, stringify_symbol

    equations = model.equations

    if eq_type == "auxiliary":
        eqs = ["{}".format(s) for s in model.symbols["auxiliaries"]]
        specs = {
            "eqs": [                                  # Input vars for aux eqns (snt3p5)
                ["exogenous", 0, "m"],
                ["states", 0, "s"],
                ["controls", 0, "x"],
                ["parameters", 0, "p"],
            ]
        }
    else:
        eqs = equations[eq_type]
        if eq_type in ("arbitrage_lb", "arbitrage_ub"):
            specs = {
                "eqs": recipes["dtcc"]["specs"]["arbitrage"]["complementarities"][
                    "left-right"                      # Get bounds for optimality conds (snt3p5)
                ]
            }
        else:
            specs = recipes["dtcc"]["specs"][eq_type]

    specs = shift_spec(specs, tshift=tshift)

    preamble_tshift = set([s[1] for s in specs["eqs"] if s[0] == "states"])
    preamble_tshift = preamble_tshift.intersection(
        set([s[1] for s in specs["eqs"] if s[0] == "controls"])  # Find common time shifts (snt3p5)
    )

    args = []
    for sg in specs["eqs"]:
        if sg[0] == "parameters":
            args.append([s for s in model.symbols["parameters"]])
        else:
            args.append([(s, sg[1]) for s in model.symbols[sg[0]]])  # Build time-shifted args (snt3p5)
    args = [[stringify_symbol(e) for e in vg] for vg in args]

    arguments = dict(zip([sg[2] for sg in specs["eqs"]], args))  # Map arg names to symbols (snt3p5)

    # temp
    eqs = [eq.split("⟂")[0].strip() for eq in eqs]   # Remove complementarity conds (snt3p5)

    if "target" in specs:
        sg = specs["target"]
        targets = [(s, sg[1]) for s in model.symbols[sg[0]]]  # Get target vars at t+shift (snt3p5)
        eqs = [eq.split("=")[1] for eq in eqs]       # Keep RHS of definitions (snt3p5)
    else:
        eqs = [
            ("({1})-({0})".format(*eq.split("=")) if "=" in eq else eq) for eq in eqs  # Convert to residual form (snt3p5)
        ]
        targets = [("out{}".format(i), 0) for i in range(len(eqs))]

    eqs = [str.strip(eq) for eq in eqs]

    eqs = [dolang.parse_string(eq) for eq in eqs]    # Convert to AST (snt3p5)
    es = Sanitizer(variables=model.variables)
    eqs = [es.transform(eq) for eq in eqs]           # Normalize var refs (snt3p5)

    eqs = [time_shift(eq, tshift) for eq in eqs]     # Apply time shift to eqns (snt3p5)

    eqs = [stringify(eq) for eq in eqs]              # Convert to canonical form (snt3p5)

    eqs = [str_expression(eq) for eq in eqs]         # Convert to evaluatable str (snt3p5)

    targets = [stringify_symbol(e) for e in targets]

    # sanitize defs ( should be )
    defs = dict()

    for k in model.definitions:
        val = model.definitions[k]
        # val = es.transform(dolang.parse_string(val))
        for t in preamble_tshift:
            s = stringify(time_shift(k, t))           # Create shifted aux var (snt3p5)
            if isinstance(val, str):
                vv = stringify(time_shift(val, t))    # Shift aux defn (snt3p5)
            else:
                vv = str(val)
            defs[s] = vv

    preamble = reorder_preamble(defs)                # Order defs by dependencies (snt3p5)

    eqs = dict(zip(targets, eqs))
    ff = FlatFunctionFactory(preamble, eqs, arguments, eq_type)  # Create optimized fn (snt3p5)

    return ff
