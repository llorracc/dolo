"""
Factory functions for model solution methods.

This module provides factory functions that construct appropriate solution methods
for different types of economic models. Key features:
- Maps model types to solution algorithms
- Handles algorithm selection and initialization
- Provides consistent interface for different solvers
- Supports custom solution method registration
"""

# plainformatter = get_ipython().display_formatter.formatters['text/plain']
# del plainformatter.type_printers[dict]

import yaml  # For YAML parsing
import numpy as np  # For numerical operations
from typing import List  # For type hints

import ast  # For Python AST manipulation
from ast import BinOp, Sub  # For AST node types

from typing import Dict  # For type hints

import dolang  # For expression handling
from dolang.grammar import str_expression  # For expression stringification
from dolang.symbolic import parse_string  # For parsing expressions
from dolang.symbolic import time_shift  # For time shifting expressions
from dolang.symbolic import Sanitizer  # For expression sanitization
from dolang.factory import FlatFunctionFactory  # For function creation


def get_name(e):  # Extract function name from expression
    """
    Extract function name from a function call expression.
    
    Parses a string containing a function call and returns the function name.
    Used to identify function names in model equations.
    
    Parameters
    ----------
    e : str
        String containing a function call expression
        
    Returns
    -------
    str
        Name of the function being called
        
    Examples
    --------
    >>> get_name('f(x, y)')
    'f'
    """
    return ast.parse(e).body[0].value.func.id  # Parse and get function identifier


def reorder_preamble(pr):  # Reorder variable definitions to respect dependencies
    """
    Reorder variable definitions to respect their dependencies.
    
    Analyzes the dependency graph of variable definitions and reorders them so that
    each variable is defined after all variables it depends on. This ensures that
    definitions can be evaluated in the correct order.
    
    Parameters
    ----------
    pr : dict
        Dictionary of variable definitions where each value may depend on
        previously defined variables
        
    Returns
    -------
    dict
        New dictionary with definitions reordered to respect dependencies
        
    Notes
    -----
    Uses a triangular solver to find a valid ordering. Raises an error if
    circular dependencies are detected.
    
    Examples
    --------
    >>> reorder_preamble({'b': 'a + 1', 'a': '2'})
    {'a': '2', 'b': 'a + 1'}
    """
    from dolang.triangular_solver import triangular_solver, get_incidence  # For dependency analysis

    inc = get_incidence(pr)  # Get dependency graph
    order = triangular_solver(inc)  # Solve for correct ordering
    d = dict()  # Initialize ordered dictionary
    prl = [*pr.items()]  # Convert to list of items
    for o in order:  # Process in dependency order
        k, v = prl[o]  # Get key-value pair
        d[k] = v  # Add to ordered dict
    return d  # Return ordered definitions


def shift_spec(specs, tshift):  # Shift timing in equation specifications
    """
    Apply time shifts to equation specifications.
    
    Modifies timing indices in equation specifications while preserving
    parameter timing. Used to shift equations forward or backward in time
    while maintaining proper variable relationships.
    
    Parameters
    ----------
    specs : dict
        Dictionary containing equation specifications with:
        - 'target': Optional target variable spec [type, timing, name]
        - 'eqs': List of equation specs [type, timing, name]
    tshift : int
        Number of periods to shift (positive for future, negative for past)
        
    Returns
    -------
    dict
        New specs dictionary with shifted timings, except for parameters
        which remain unchanged
    """
    ss = dict()  # Initialize shifted specs
    if "target" in specs:  # Handle target if present
        e = specs["target"]  # Get target spec
        ss["target"] = [e[0], e[1] + tshift, e[2]]  # Shift target timing

    ss["eqs"] = [  # Process equations
        ([e[0], e[1] + tshift, e[2]] if e[0] != "parameters" else e)  # Shift non-parameters
        for e in specs["eqs"]  # For each equation spec
    ]
    return ss  # Return shifted specs


def get_factory(model, eq_type: str, tshift: int = 0):  # Create function factory for equation type
    """
    Create a function factory for generating model equation evaluators.
    
    Constructs a factory that can create functions to evaluate specific types of
    model equations (e.g., transition equations, arbitrage conditions). Handles:
    - Variable timing and shifts
    - Equation preprocessing and sanitization
    - Definition dependencies
    - Complementarity conditions
    
    Parameters
    ----------
    model : Model
        The model containing equations to process
    eq_type : str
        Type of equations to generate evaluator for:
        - 'transition': State transition equations
        - 'arbitrage': Optimality conditions
        - 'auxiliary': Helper equations
        - 'arbitrage_lb'/'arbitrage_ub': Bound constraints
    tshift : int, default=0
        Time shift to apply to equations
        
    Returns
    -------
    FlatFunctionFactory
        Factory object that can create equation evaluation functions
    """

    from dolo.compiler.model import decode_complementarity  # For handling constraints
    from dolo.compiler.recipes import recipes  # For model recipes
    from dolang.symbolic import stringify, stringify_symbol  # For expression conversion

    equations = model.equations  # Get model equations

    if eq_type == "auxiliary":  # Handle auxiliary equations
        eqs = ["{}".format(s) for s in model.symbols["auxiliaries"]]  # Format auxiliary symbols
        specs = {  # Define auxiliary specs
            "eqs": [
                ["exogenous", 0, "m"],  # Current exogenous
                ["states", 0, "s"],  # Current states
                ["controls", 0, "x"],  # Current controls
                ["parameters", 0, "p"],  # Parameters
            ]
        }
    else:  # Handle other equation types
        eqs = equations[eq_type]  # Get equations of specified type
        if eq_type in ("arbitrage_lb", "arbitrage_ub"):  # Handle bounds
            specs = {  # Get complementarity specs
                "eqs": recipes["dtcc"]["specs"]["arbitrage"]["complementarities"][
                    "left-right"
                ]
            }
        else:
            specs = recipes["dtcc"]["specs"][eq_type]  # Get standard specs

    specs = shift_spec(specs, tshift=tshift)  # Apply time shift to specs

    preamble_tshift = set([s[1] for s in specs["eqs"] if s[0] == "states"])  # Get state shifts
    preamble_tshift = preamble_tshift.intersection(  # Find common shifts
        set([s[1] for s in specs["eqs"] if s[0] == "controls"])  # With control shifts
    )

    args = []  # Initialize argument list
    for sg in specs["eqs"]:  # Process each equation spec
        if sg[0] == "parameters":  # Handle parameters
            args.append([s for s in model.symbols["parameters"]])  # Add parameter symbols
        else:  # Handle other variables
            args.append([(s, sg[1]) for s in model.symbols[sg[0]]])  # Add with timing
    args = [[stringify_symbol(e) for e in vg] for vg in args]  # Convert to strings

    arguments = dict(zip([sg[2] for sg in specs["eqs"]], args))  # Create arguments dict

    eqs = [eq.split("⟂")[0].strip() for eq in eqs]  # Remove complementarity conditions

    if "target" in specs:  # Handle equations with targets
        sg = specs["target"]  # Get target spec
        targets = [(s, sg[1]) for s in model.symbols[sg[0]]]  # Create target list
        eqs = [eq.split("=")[1] for eq in eqs]  # Keep RHS of equations
    else:  # Handle equations without targets
        eqs = [  # Convert to residual form
            ("({1})-({0})".format(*eq.split("=")) if "=" in eq else eq) for eq in eqs
        ]
        targets = [("out{}".format(i), 0) for i in range(len(eqs))]  # Create dummy targets

    eqs = [str.strip(eq) for eq in eqs]  # Clean equations

    eqs = [dolang.parse_string(eq) for eq in eqs]  # Parse equations
    es = Sanitizer(variables=model.variables)  # Create sanitizer
    eqs = [es.transform(eq) for eq in eqs]  # Sanitize equations

    eqs = [time_shift(eq, tshift) for eq in eqs]  # Apply time shifts

    eqs = [stringify(eq) for eq in eqs]  # Convert to strings

    eqs = [str_expression(eq) for eq in eqs]  # Format expressions

    targets = [stringify_symbol(e) for e in targets]  # Convert targets to strings

    defs = dict()  # Initialize definitions

    for k in model.definitions:  # Process model definitions
        val = model.definitions[k]  # Get definition value
        for t in preamble_tshift:  # For each time shift
            s = stringify(time_shift(k, t))  # Shift and stringify key
            if isinstance(val, str):  # Handle string values
                vv = stringify(time_shift(val, t))  # Shift and stringify
            else:  # Handle other values
                vv = str(val)  # Convert to string
            defs[s] = vv  # Add to definitions

    preamble = reorder_preamble(defs)  # Order definitions by dependencies

    eqs = dict(zip(targets, eqs))  # Create equations dictionary
    ff = FlatFunctionFactory(preamble, eqs, arguments, eq_type)  # Create function factory

    return ff  # Return factory
