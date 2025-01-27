"""
Recipe system for model equation block processing in dolo.

This module implements a recipe system for handling model equations, with key features:
- Recursive block processing [default]: Equations are processed sequentially
- Optional left-hand side ordering [default]: Equations can be reordered
- Cross-block dependencies: Manages relationships between equation blocks
- Substitutable dummy blocks: Blocks that are substituted into other equations

The recipe system provides a flexible way to specify how different types of
model blocks should be processed and solved.
"""

import os, yaml, sys  # Core modules for file/system operations and YAML parsing

if getattr(sys, "frozen", False):  # Check if running as PyInstaller executable
    # we are running in a |PyInstaller| bundle
    DIR_PATH = sys._MEIPASS  # Get bundle directory path
else:
    DIR_PATH, this_filename = os.path.split(__file__)  # Get module directory path

DATA_PATH = os.path.join(DIR_PATH, "recipes.yaml")  # Path to model recipe definitions

with open(DATA_PATH, "rt", encoding="utf-8") as f:  # Load recipe file with UTF-8 encoding
    recipes = yaml.safe_load(f)  # Parse YAML into recipe dictionary
