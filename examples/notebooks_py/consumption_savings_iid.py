#!/usr/bin/env python
# coding: utf-8

# %%
import matplotlib.pyplot as plt
from pathlib import Path
from dolo import *

# Simple path setup that works for both notebook and script
examples_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd().parent
model = yaml_import(str(examples_dir / "models" / "consumption_savings_iid.yaml"))

# %%
# Time iteration
dr = time_iteration(model)

# %%
# Create simulation before plotting it
sim = simulate(model, dr, i0=1, N=100)

# %%
plt.subplot(121)
for i in range(10):
    plt.plot(sim.sel(N=i,V='c'), color='red', alpha=0.5)
plt.ylabel("$c_t$")
plt.xlabel("$t$")
plt.subplot(122)
for i in range(10):
    plt.plot(sim.sel(N=i,V='w'), color='red', alpha=0.5)
plt.xlabel("$t$")
plt.ylabel("$w_t$")

plt.tight_layout()

# %% [markdown]
"""
## Ergodic distribution
"""

# %%
sim_long = simulate(model, dr, i0=1, N=1000, T=200)

# %%
import seaborn
seaborn.distplot(sim_long.sel(T=199, V='w'))
plt.xlabel("$w$")

# %% [markdown]
"""
## Plotting Decision Rule
"""

# %%
tab = tabulate(model, dr,'w')

# %%
stable_wealth = model.eval_formula('1/r+(1-1/r)*w(0)', tab)
plt.plot(tab['w'], tab['w'],color='black', linestyle='--')
plt.plot(tab['w'], stable_wealth,color='black', linestyle='--')
plt.plot(tab['w'], tab['c'])
plt.xlabel("w_t")
plt.ylabel("c_t")
plt.grid()

