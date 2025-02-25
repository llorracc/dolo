#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from pathlib import Path
from dolo import *
from dolo.algos.egm import egm
import numpy as np

# Simple path setup that works for both notebook and script
examples_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd().parent

# Load the EGM version of the model
model = yaml_import(str(examples_dir / "models" / "consumption_savings_iid_egm.yaml"))

# Set up asset grid for the EGM algorithm
a_grid = np.linspace(0.1, 10, 10)**2

# Solve the model using EGM, limiting to one iteration (one period backward solve) using maxit=1
sol = egm(model, a_grid=a_grid, maxit=1, verbose=True)

# Output the solution summary
print("EGM solution computed with maxit=1")
print(sol) 