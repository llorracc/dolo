#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
from pathlib import Path
from dolo import *
from dolo.algos.egm import egm

# Simple path setup that works for both notebook and script
examples_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd().parent
model = yaml_import(str(examples_dir / "models" / "consumption_savings_iid_egm.yaml"))


# In[3]:

import numpy as np


from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.decision_rule import CallableDecisionRule

# Option 1: Simple function wrapper
def return_the_state(i, s):
    """Function to implement c=w rule
    i: exogenous state index (ignored for c=w rule)
    s: endogenous state value (wealth)
    """
    return s  # Return state directly

# Create decision rule by directly passing the function
c_eq_w = CallableDecisionRule(return_the_state)

sol = egm(model, a_grid=np.linspace(0.0, 10, 10) ** 2, dr0=c_eq_w, maxit=2)

dr = sol.dr  # Extract decision rule from EGMResult object

exit()
# stable_wealth = model.eval_formula('1/r+(1-1/r)*w(0)', tab)
plt.plot(tab['w'], tab['w'],color='black', linestyle='--')
#plt.plot(tab['w'], stable_wealth,color='black', linestyle='--')
plt.plot(tab['w'], tab['c'])
plt.xlabel("w_t")
plt.ylabel("c_t")
plt.grid()
plt.show(block=True)
