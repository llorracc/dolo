#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from pathlib import Path
from dolo import *
import numpy as np
# Import our custom function
import sys
# Get the path to the root dolo directory
script_path = Path(__file__).resolve()  # Full path to this script
dolo_root = script_path.parent.parent.parent  # Go up three levels: notebooks_py -> examples -> dolo
sys.path.append(str(dolo_root))
from dolo_custom.algos import solve_back_from_successor
from dolo.numeric.decision_rule import CallableDecisionRule, DecisionRule

# Turn off interactive mode - we'll use a different approach to keep the figure open
plt.ioff()

# Simple path setup that works for both notebook and script
examples_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd().parent
model = yaml_import(str(examples_dir / "models" / "consumption_savings_iid.yaml"))

# Print model calibration to verify parameters
print("Model Parameters:")
param_names = model.symbols['parameters']
param_values = model.calibration['parameters']
for i, name in enumerate(param_names):
    print(f"  {name} = {param_values[i]}")

# Define the wealth grid for all methods
a_grid = np.linspace(0.1, 10, 50) ** 2  # Using more points for better visualization

# Use None as successor (will use the default StateEqualsControlRule)
terminal_rule = None

print("Solving using two different solution methods, going back one period from T...")

# 1. Solve using EGM
print("\n1. Solving using EGM method...")
sol_egm = solve_back_from_successor(
    model=model,
    method='egm',
    successor=terminal_rule,
    a_grid=a_grid,
    verbose=True,
    maxit=1  # Only one iteration to go from T to T-1
)
dr_egm = sol_egm.dr

# 2. Solve using Time Iteration
print("\n2. Solving using Time Iteration method...")
sol_time = solve_back_from_successor(
    model=model,
    method='time_iteration',
    successor=terminal_rule,
    verbose=True,
    maxit=1  # Only one iteration to go from T to T-1
)
dr_time = sol_time.dr

# Tabulate decision rules
print("\nTabulating decision rules...")
tab_egm = tabulate(model, dr_egm, 'w')
tab_time = tabulate(model, dr_time, 'w')

# Print grid points to check if they're the same
print(f"\nEGM grid points: {tab_egm['w'].min():.4f} to {tab_egm['w'].max():.4f}, {len(tab_egm['w'])} points")
print(f"Time Iteration grid points: {tab_time['w'].min():.4f} to {tab_time['w'].max():.4f}, {len(tab_time['w'])} points")

# Create a unified wealth grid for comparison
w_min = min(tab_egm['w'].min(), tab_time['w'].min())
w_max = max(tab_egm['w'].max(), tab_time['w'].max())
w_unified = np.linspace(w_min, w_max, 100)

# Interpolate all decision rules onto the unified grid
from scipy.interpolate import interp1d

egm_interp = interp1d(tab_egm['w'], tab_egm['c'], bounds_error=False, fill_value="extrapolate")
time_interp = interp1d(tab_time['w'], tab_time['c'], bounds_error=False, fill_value="extrapolate")

c_egm = egm_interp(w_unified)
c_time = time_interp(w_unified)

# Calculate mean solution
c_mean = (c_egm + c_time) / 2

# Create comparison plots
fig, axes = plt.subplots(2, 1, figsize=(15, 10), num="Solution Comparison (T-1)")

# Plot 1: Decision rules
axes[0].plot(w_unified, c_egm, 'b-', label='EGM')
axes[0].plot(w_unified, c_time, 'r-', label='Time Iteration')
axes[0].plot(w_unified, w_unified, 'k--', alpha=0.5, label='45-degree line')
axes[0].set_xlabel('Wealth (w)')
axes[0].set_ylabel('Consumption (c)')
axes[0].set_title('Comparison of Decision Rules (T-1 solution)')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Differences from mean
axes[1].plot(w_unified, c_egm - c_mean, 'b-', label='EGM - Mean')
axes[1].plot(w_unified, c_time - c_mean, 'r-', label='Time Iteration - Mean')
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Wealth (w)')
axes[1].set_ylabel('Difference from Mean')
axes[1].set_title('Differences Between Solution Methods')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('solution_comparison.png')
print("\nComparison plot saved as 'solution_comparison.png'")

# Calculate some statistics about the differences
max_diff = np.max(np.abs(c_egm - c_time))
mean_diff = np.mean(np.abs(c_egm - c_time))
max_diff_egm = np.max(np.abs(c_egm - c_mean))
max_diff_time = np.max(np.abs(c_time - c_mean))

print(f"\nDifference statistics:")
print(f"Maximum absolute difference between methods: {max_diff:.6f}")
print(f"Mean absolute difference between methods: {mean_diff:.6f}")
print(f"Maximum difference from mean - EGM: {max_diff_egm:.6f}")
print(f"Maximum difference from mean - Time Iteration: {max_diff_time:.6f}")

# Display the figure and block execution (this will keep the figure open)
print("\nDisplaying plot. Close the window to exit.")
plt.show()

# This line will only execute after the plot window is closed
print("Plot window was closed. Exiting...")

# ## Stochastic Simulations

# In[4]:


# Shocks are discretized as a markov chain by default:

# Shocks are discretized as a markov chain by default:
dp = model.exogenous.discretize()
sim_shock = dp.simulate(10, 100, i0=1)
for i in range(10):
    plt.plot(sim_shock[:,i,0], color='red', alpha=0.5)
plt.ioff()  # Turn off interactive mode

# In[5]:


dr = sol.dr  # Extract decision rule from EGMResult object
sim = simulate(model, dr, i0=1, N=100)


# In[6]:


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


# ## Ergodic distribution

# In[7]:


sim_long = simulate(model, dr, i0=1, N=1000, T=200)


# In[8]:


import seaborn
seaborn.distplot(sim_long.sel(T=199, V='w'))
plt.xlabel("$w$")


# ## Plotting Decision Rule

# In[9]:


tab = tabulate(model, dr,'w')


# In[10]:


stable_wealth = model.eval_formula('1/r+(1-1/r)*w(0)', tab)
plt.plot(tab['w'], tab['w'],color='black', linestyle='--')
plt.plot(tab['w'], stable_wealth,color='black', linestyle='--')
plt.plot(tab['w'], tab['c'])
plt.xlabel("w_t")
plt.ylabel("c_t")
plt.grid()

print("Displaying plot. Close the plot window manually to continue.")
plt.show(block=True)

