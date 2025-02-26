import importlib
import importlib.util
import sys
import inspect
import os
import numpy as np
from dolo.compiler.model import Model
from dolo.numeric.decision_rule import CallableDecisionRule, DecisionRule

def solve_back_from_successor(model: Model, method='time_iteration', successor=None, solver_path=None, **kwargs):
    """
    Solves a model backward starting from a provided successor solution or rule.
    
    This function standardizes how different Dolo solvers handle successor solutions,
    ensuring they all treat the provided successor consistently regardless of their 
    default infinite-horizon assumptions.
    
    Parameters:
    -----------
    model : Model
        The dolo model to solve
    method : str
        Solution method to use (e.g., 'time_iteration', 'egm', 'value_iteration')
    successor : DecisionRule, object, or None
        Decision rule or solution object for the subsequent period. 
        If it's a solution object, the function will extract the decision rule.
        If None, defaults to a rule where control = state 
        (e.g., consumption = wealth in consumption-savings models)
    solver_path : str, optional
        Path to a directory or specific Python file containing the solver
        Can be absolute or relative path
    **kwargs : dict
        Additional arguments to pass to the solution method
        
    Returns:
    --------
    solution : object
        The solution produced by stepping back from the successor
    
    Notes:
    ------
    This function normalizes behavior across solvers by setting appropriate iteration 
    parameters. Specifically, it sets maxit=2 for EGM and maxit=1 for other solvers
    when using a successor, unless overridden in kwargs.
    """
    # Copy kwargs to avoid modifying the original
    kwargs_copy = kwargs.copy()
    
    # Process the successor argument
    successor_dr = None
    
    if successor is None:
        # Create default successor rule (control = state) if none provided
        class StateEqualsControlRule(CallableDecisionRule):
            def eval_s(self, s):
                """Return the state as the control value (e.g., c=w)"""
                return s
                
            def eval_is(self, i, s):
                """Return the state as the control value for any index"""
                return s
                
            def eval_ms(self, m, s):
                """Return the state as the control value for any exogenous value"""
                return s
        
        successor_dr = StateEqualsControlRule()
    
    elif isinstance(successor, DecisionRule):
        # If successor is already a decision rule, use it directly
        successor_dr = successor
    
    else:
        # Try to extract a decision rule from the successor
        try:
            if hasattr(successor, 'dr'):
                successor_dr = successor.dr
            # Add additional extractors for other types of successors if needed
            else:
                raise ValueError("Could not extract a decision rule from the provided successor")
        except:
            raise ValueError("Successor must be a DecisionRule, a solution object with a 'dr' attribute, or None")
    
    # Try to import the solver
    solver_func = None
    
    # Case 1: User provided a specific path to the solver
    if solver_path is not None:
        # Normalize the path
        solver_path = os.path.normpath(os.path.expanduser(solver_path))
        
        # Check if it's a .py file or a directory
        if solver_path.endswith('.py'):
            # It's a Python file
            module_path = solver_path
            module_name = os.path.basename(solver_path)[:-3]  # Remove .py extension
        else:
            # It's a directory, construct the full path to the Python file
            module_path = os.path.join(solver_path, f"{method}.py")
            module_name = method
            
            # If that exact path doesn't exist, try with underscores (common convention)
            if not os.path.exists(module_path):
                underscore_name = method.replace('-', '_')
                module_path = os.path.join(solver_path, f"{underscore_name}.py")
                module_name = underscore_name
        
        # Try to import the module from the file path
        if os.path.exists(module_path):
            try:
                # Use importlib.util for importing from file path
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for the solver function
                    if hasattr(module, module_name):
                        solver_func = getattr(module, module_name)
                    elif hasattr(module, method):
                        solver_func = getattr(module, method)
                    else:
                        # Search for any callable that might be the solver
                        for name, obj in inspect.getmembers(module):
                            if callable(obj) and not name.startswith('_'):
                                solver_func = obj
                                break
            except Exception as e:
                raise ImportError(f"Error importing solver from {module_path}: {str(e)}")
    
    # Case 2: No path provided, try standard dolo locations
    if solver_func is None:
        # Possible locations for solver modules
        possible_modules = [
            f"dolo.algos.{method}",           # Standard dolo location
            f"{method}",                       # Direct import if it's in path
            f"custom_dolo.algos.{method}"     # Example of user custom location
        ]
        
        # Try to import the solver from possible locations
        for module_name in possible_modules:
            try:
                # Try to import the module
                module = importlib.import_module(module_name)
                
                # If it's a direct module (like dolo.algos.egm), look for the function with same name
                if hasattr(module, method):
                    solver_func = getattr(module, method)
                    break
                    
                # If it's another form, look for a callable that might be the solver function
                for name, obj in inspect.getmembers(module):
                    if callable(obj) and name.lower() == method.lower():
                        solver_func = obj
                        break
                
                # If we found a solver, break out of the module search loop
                if solver_func is not None:
                    break
                    
            except (ImportError, ModuleNotFoundError):
                # Module not found, try next location
                continue
    
    # If no solver found, raise error
    if solver_func is None:
        if solver_path:
            raise ValueError(f"No solver named '{method}' found at location '{solver_path}'")
        else:
            raise ValueError(f"No solver named '{method}' found in dolo.algos or user extensions.")
    
    # Special handling for EGM when successor is provided
    # Only set these defaults if user didn't explicitly provide maxit
    if 'maxit' not in kwargs_copy:
        if method.lower() == 'egm':
            # EGM needs maxit=2 to get one true backward iteration from successor
            kwargs_copy['maxit'] = 2
        else:
            # Other solvers just need maxit=1
            kwargs_copy['maxit'] = 1
    
    # Pass everything to the appropriate solver, using successor_dr as dr0
    return solver_func(model, dr0=successor_dr, **kwargs_copy) 