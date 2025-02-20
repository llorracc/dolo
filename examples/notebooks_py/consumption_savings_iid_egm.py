#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from pathlib import Path
from dolo import *
from dolo.algos.egm import egm

# Simple path setup that works for both notebook and script
examples_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd().parent
model = yaml_import(str(examples_dir / "models" / "consumption_savings_iid.yaml"))


# In[2]:


model = yaml_import("../models/consumption_savings_iid_egm.yaml")


# In[3]:


import numpy as np
sol = egm(model, a_grid=np.linspace(0.1, 10, 10) ** 2, verbose=True)


# One can also try the faster version

# ## Stochastic Simulations

# In[4]:


# Shocks are discretized as a markov chain by default:

# Shocks are discretized as a markov chain by default:
dp = model.exogenous.discretize()
sim_shock = dp.simulate(10, 100, i0=1)
for i in range(10):
    plt.plot(sim_shock[:,i,0], color='red', alpha=0.5)


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

