"""
Miscellaneous utilities for model compilation and calibration.

This module provides helper classes and functions for:
- Managing model calibration data
- Converting between different variable representations
- Handling model symbols and equations
- Supporting the model compilation process
"""

import numpy as np  # For numerical operations
from typing import Dict, List, Set, TypeVar, Tuple  # For type hints
from dolang.symbolic import str_expression  # For symbolic manipulation
from dataclasses import dataclass  # For data classes


def calibration_to_vector(symbols, calibration_dict):  # Convert calibration dict to vectors
    """
    Convert a calibration dictionary to vectorized format.
    
    Takes a calibration dictionary with symbol-value pairs and converts it to
    a dictionary of numpy arrays organized by symbol groups.
    
    Parameters
    ----------
    symbols : dict
        Dictionary mapping symbol groups to lists of symbol names
    calibration_dict : dict
        Dictionary mapping symbol names to calibrated values
        
    Returns
    -------
    dict
        Dictionary mapping symbol groups to numpy arrays of calibrated values
        
    Notes
    -----
    Uses triangular solver to handle interdependent calibration definitions.
    Missing values are filled with numpy.nan.
    """
    from dolang.triangular_solver import solve_triangular_system  # For solving calibration system
    from numpy import nan  # For handling missing values

    sol = solve_triangular_system(calibration_dict)  # Solve for calibration values

    calibration = dict()  # Initialize output dictionary
    for group in symbols:  # Process each symbol group
        t = np.array([sol.get(s, nan) for s in symbols[group]], dtype=float)  # Convert to array
        calibration[group] = t  # Store group values

    return calibration  # Return vectorized calibration


def calibration_to_dict(symbols, calib):  # Convert calibration vectors to dict

    if not isinstance(symbols, dict):  # Handle non-dict symbols
        symbols = symbols.symbols  # Get symbols dictionary

    d = dict()  # Initialize output dictionary
    for group, values in calib.items():  # Process each group
        if group == "covariances":  # Skip covariance matrix
            continue
        syms = symbols[group]  # Get group symbols
        for i, s in enumerate(syms):  # Process each symbol
            d[s] = values[i]  # Store calibrated value

    return d  # Return calibration dictionary


from dolo.compiler.misc import calibration_to_dict  # For calibration conversion

import copy  # For deep copying

equivalent_symbols = dict(actions="controls")  # Define symbol equivalences


class LoosyDict(dict):  # Dictionary with flexible key matching
    """
    Dictionary subclass that allows lookup using equivalent keys.
    
    Extends standard dictionary to support looking up values using alternative
    key names defined in an equivalence mapping. If a key is not found directly,
    checks for any equivalent keys before raising KeyError.
    
    Parameters
    ----------
    *args
        Standard dictionary initialization arguments
    equivalences : dict, optional
        Dictionary mapping primary keys to lists of equivalent keys
    **kwargs
        Additional keyword arguments for dictionary initialization
        
    Examples
    --------
    >>> d = LoosyDict({'x': 1}, equivalences={'x': ['y', 'z']})
    >>> d['y']  # Returns 1 since 'y' is equivalent to 'x'
    1
    """
    def __init__(self, *args, equivalences=None, **kwargs):  # Initialize with optional equivalences
        super().__init__(*args, **kwargs)  # Initialize base dictionary
        if equivalences is None:  # Handle default equivalences
            self.equivalences = {}  # Empty equivalence map
        else:
            self.equivalences = equivalences  # Store equivalence mappings

    def __getitem__(self, item):  # Custom dictionary lookup
        try:
            return super().__getitem__(item)  # Try direct lookup
        except KeyError as e:  # Handle key not found
            for k in self.equivalences.keys():  # Check equivalences
                if item in self.equivalences[k]:  # If item has equivalent
                    return super().__getitem__(k)  # Return equivalent value
            raise e  # No equivalent found


class CalibrationDict:  # Dictionary for model calibration
    def __init__(self, symbols, calib, equivalences=equivalent_symbols):  # Initialize calibration
        calib = copy.deepcopy(calib)  # Make deep copy of calibration
        for v in calib.values():  # Make arrays read-only
            v.setflags(write=False)  # Prevent modification
        self.symbols = symbols  # Store symbol definitions
        self.flat = calibration_to_dict(symbols, calib)  # Create flat dictionary
        self.grouped = LoosyDict(  # Create grouped dictionary
            **{k: v for (k, v) in calib.items()}, equivalences=equivalences
        )

    def __getitem__(self, p):  # Custom dictionary lookup
        if isinstance(p, tuple):  # Handle multiple keys
            return [self[e] for e in p]  # Return list of values
        if p in self.symbols.keys() or (p in self.grouped.__equivalences__.keys()):  # Check group keys
            return self.grouped[p]  # Return group values
        else:
            return self.flat[p]  # Return single value


def allocating_function(inplace_function, size_output):  # Wrap inplace function
    def new_function(*args, **kwargs):  # Create allocating wrapper
        val = np.zeros(size_output)  # Allocate output array
        nargs = args + (val,)  # Add output array to arguments
        inplace_function(*nargs)  # Call inplace function
        if "diff" in kwargs:  # Handle differentiation
            return numdiff(new_function, args)  # Compute numerical derivatives
        return val  # Return computed value

    return new_function  # Return wrapped function


def numdiff(fun, args):  # Compute numerical derivatives
    """
    Compute numerical derivatives using finite differences.
    
    Implements vectorized numerical differentiation for functions with multiple
    inputs and outputs. Uses central differences with a small epsilon step.
    
    Parameters
    ----------
    fun : callable
        Function to differentiate, should accept and return numpy arrays
    args : list
        List of input arrays to differentiate with respect to
        
    Returns
    -------
    list
        List containing:
        - First element: Function value at input point
        - Remaining elements: Jacobian matrices for each input argument
        
    Notes
    -----
    Uses forward differences with step size epsilon=1e-8. For better accuracy
    in some cases, central differences could be used instead.
    """

    epsilon = 1e-8  # Step size for differentiation
    args = list(args)  # Convert args to list
    v0 = fun(*args)  # Compute base value
    N = v0.shape[0]  # Get number of points
    l_v = len(v0)  # Get output dimension
    dvs = []  # Initialize derivative list
    for i, a in enumerate(args):  # Loop over arguments
        l_a = (a).shape[1]  # Get argument dimension
        dv = np.zeros((N, l_v, l_a))  # Initialize derivatives
        nargs = list(args)  # Copy arguments
        for j in range(l_a):  # Loop over components
            xx = args[i].copy()  # Copy current argument
            xx[:, j] += epsilon  # Perturb component
            nargs[i] = xx  # Update arguments
            dv[:, :, j] = (fun(*nargs) - v0) / epsilon  # Compute derivative
        dvs.append(dv)  # Store derivatives
    return [v0] + dvs  # Return value and derivatives


def equivalent_symbols(group):  # Define symbol equivalences for a group
    """
    Get list of equivalent symbol names for a given group.
    
    Provides alternative names that can be used interchangeably for certain
    groups of model variables. This allows for more flexible model specification
    by supporting multiple conventional naming schemes.
    
    Parameters
    ----------
    group : str
        Name of symbol group to get equivalences for
        
    Returns
    -------
    list
        List of equivalent names for the group:
        - For 'states': ['states', 'predetermined_variables']
        - For 'parameters': ['parameters', 'shocks']
        - For others: [group] (no equivalences)
    """
    if group == "states":  # Handle state variables
        return ["states", "predetermined_variables"]  # Equivalent names for states
    elif group == "parameters":  # Handle parameters
        return ["parameters", "shocks"]  # Equivalent names for parameters
    else:
        return [group]  # No equivalences for other groups


def timeshift(expr, variables, shift):  # Apply time shift to expression
    """
    Apply a time shift to variables in a symbolic expression.
    
    Modifies timing of variables in an expression by adding or subtracting periods.
    For example, can convert x(t) to x(t+1) or x(t-1).
    
    Parameters
    ----------
    expr : str or Expression
        The symbolic expression to time shift
    variables : list
        List of variable names that can be shifted
    shift : int
        Number of periods to shift (positive for future, negative for past)
        
    Returns
    -------
    str
        Time-shifted expression as a string
        
    Examples
    --------
    >>> timeshift('x + y(1)', ['x', 'y'], 1)
    'x(1) + y(2)'
    """
    if shift == 0:  # No shift needed
        return expr  # Return unchanged
    from dolang.symbolic import stringify  # Import string conversion
    return stringify(timeshift_implementation(expr, variables, shift))  # Convert and return


def timeshift_implementation(expr, variables, shift):  # Implementation of time shifting
    """
    Internal implementation of time shifting for symbolic expressions.
    
    Recursively processes expression trees to apply time shifts to variables.
    Handles both string expressions and parsed symbolic expression trees.
    
    Parameters
    ----------
    expr : str or Expression
        Expression to time shift, either as string or parsed tree
    variables : list
        List of variable names that can be shifted
    shift : int
        Number of periods to shift
        
    Returns
    -------
    Expression
        Time-shifted expression tree
        
    Notes
    -----
    This is an internal function used by timeshift(). It handles the recursive
    traversal of expression trees and actual shifting of time indices.
    """
    from dolang.symbolic import parse_string  # For parsing expressions
    from dolang.symbolic import Symbol, TSymbol  # Symbol types

    if isinstance(expr, str):  # Handle string input
        return timeshift_implementation(parse_string(expr), variables, shift)  # Parse and shift

    if isinstance(expr, Symbol):  # Handle basic symbols
        if expr.name in variables:  # If symbol is a variable
            return TSymbol(expr.name, shift)  # Apply time shift
        else:
            return expr  # Leave unchanged
    elif isinstance(expr, TSymbol):  # Handle time-indexed symbols
        if expr.name in variables:  # If symbol is a variable
            return TSymbol(expr.name, expr.date + shift)  # Add shift to date
        else:
            return expr  # Leave unchanged
    else:  # Handle other expression types
        return expr.__class__(*[timeshift_implementation(e, variables, shift) for e in expr.children])  # Recursively shift


def normalize(s):  # Normalize string representation
    """
    Normalize string representation of a symbol or equation.
    
    Standardizes the string format by removing extra whitespace and
    applying consistent formatting rules.
    
    Parameters
    ----------
    s : str
        String to normalize
        
    Returns
    -------
    str
        Normalized string representation
    """
    s = str(s)  # Convert to string
    s = s.strip()  # Remove whitespace
    return s  # Return normalized string


class CalibrationDict(dict):  # Dictionary for model calibration
    """
    Dictionary subclass for storing model calibration data.
    
    Extends dict to provide specialized handling of calibration data
    including symbol groups and value access patterns.
    
    Parameters
    ----------
    symbols : dict
        Dictionary mapping symbol groups to symbol lists
    calibration : dict
        Dictionary mapping symbols to calibrated values
    """
