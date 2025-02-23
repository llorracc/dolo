# from __future__ import print_function

# This module is supposed to be imported first
# it contains global variables used for configuration


# try to register printing methods if IPython is running

# Global configuration settings for dolo package

# Configuration flags
save_plots = False                                   # Whether to save plots vs display (snt3p5)
real_type = "double"                                 # Float precision for numerics (snt3p5)
debug = False                                        # Enable debug logging/checks (snt3p5)

import warnings

from dolo.misc.termcolor import colored              # Terminal output formatting (snt3p5)


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """Format warning messages in a single line with color highlighting"""
    return "{}:{}:{}\n    {}\n".format(
        colored(category.__name__, "yellow"), filename, lineno, message
    )


warnings.formatwarning = warning_on_one_line         # Use custom warning format (snt3p5)

# Configure numpy to ignore all warnings
import numpy

numpy.seterr(all="ignore")                          # Suppress numeric warnings (snt3p5)


# Set up temporary directory for compiled functions
import tempfile, sys

temp_dir = tempfile.mkdtemp(prefix="dolo_")         # Create dir for JIT code (snt3p5)
sys.path.append(temp_dir)                           # Allow importing compiled fns (snt3p5)


# from IPython.core.display import display
