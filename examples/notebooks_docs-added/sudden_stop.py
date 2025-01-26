#!/usr/bin/env python
# coding: utf-8

# # Sudden Stop Model
# 
# In this notebook we replicate the baseline model exposed in 
# 
# `From Sudden Stops to Fisherian Deflation, Quantitative Theory and Policy` by __Anton Korinek and Enrique G. Mendoza__
# 
# The file `sudden_stop.yaml` which is printed below, describes the model, and must be included in the same directory as this notebook.

# ## importing necessary functions

# In[7]:


from matplotlib import pyplot as plt
import numpy as np


# In[8]:


from dolo import *
from dolo.algos import time_iteration
from dolo.algos import plot_decision_rule, simulate


# ## writing the model

# In[9]:


pwd


# In[10]:


# filename = 'https://raw.githubusercontent.com/EconForge/dolo/master/examples/models/compat/sudden_stop.yaml'
filename = '../models/sudden_stop.yaml'
# the model file is coded in a separate file called sudden_stop.yaml
# note how the borrowing constraint is implemented as complementarity condition
pcat(filename)


# ## importing the model
# 
# Note, that residuals, are not zero at the calibration we supply. This is because the representative agent is impatient
# and we have $\beta<1/R$. In this case it doesn't matter.
# 
# By default, the calibrated value for endogenous variables are used as a (constant) starting point for the decision rules.

# In[11]:


model = yaml_import(filename)
model


# In[72]:


# to avoid numerical glitches we choose a relatively high number of grid points
mdr = time_iteration(model, verbose=True)


# In[75]:


# produce the plots
n_steps = 100

tab0 = tabulate(model, mdr, 'l', i0=0, n_steps=n_steps)
tab1 = tabulate(model, mdr, 'l', i0=1, n_steps=n_steps)


# In[76]:


lam_inf = model.calibration['lam_inf']


# In[77]:


plt.subplot(121)

plt.plot(tab0['l'], tab0['l'], linestyle='--', color='black', label='')
plt.plot(tab0['l'], lam_inf*tab0['c'], linestyle='--', color='black', label='')
plt.plot(tab0['l'], tab0['b'], label='$b_t$ (bad state)' )
plt.plot(tab0['l'], tab1['b'], label='$b_t$ (good state)')
# plt.plot(tab0['l'], lam_inf*tab1['c'])
plt.grid()
plt.xlabel('$l_t$')

plt.legend(loc= 'upper left')

plt.subplot(122)
plt.plot(tab0['l'], tab0['c'], label='$c_t$ (bad state)' )
plt.plot(tab0['l'], tab1['c'], label='$c_t$ (good state)' )
plt.legend(loc= 'lower right')
plt.grid()
plt.xlabel('$l_t$')

plt.suptitle("Decision Rules")


# In[84]:


# if we want we can use altair/vega-lite instead


# In[86]:


import altair as alt
import pandas


# In[166]:


# first we need to convert data into flat format
df = pandas.concat([tab0, tab1], keys=['bad','good'], names=['experiment'])
df = df.reset_index().drop(columns=['level_1']) # maybe there is a more elegant option here


# In[176]:


lam_inf = model.calibration['lam_inf']


# In[208]:


# then we can play
base = alt.Chart(df).mark_line()
ch1 = base.encode(x='l', y='b', color='experiment').interactive()
ch1_min = base.mark_line(color='grey').transform_calculate(minc=f'{lam_inf}*datum.c').encode(x='l',y=alt.Y('minc:Q', aggregate='min'))
ch1_max = base.mark_line(color='grey').encode(x='l',y='l')
ch2 = base.encode(x='l', y='c', color='experiment')

(ch1_min+ch1_max+ch1|ch2)


# ## stochastic simulations

# In[78]:


i_0 = 1 # we start from the good state
sim = simulate(model, mdr, i0=i_0, s0=np.array([0.5]), N=1, T=100) # markov_indices=markov_indices)
sim # result is an xarray object


# In[79]:


plt.subplot(211)
plt.plot(sim.sel(V='y'))
plt.subplot(212)
plt.plot(sim.sel(V='b'))


# ## Sensitivity analysis
# 
# Here we want to compare the saving behaviour as a function of risk aversion $\sigma$.
# We contrast the baseline $\sigma=2$ with the high aversion scenario $\sigma=16$.

# In[80]:


# we solve the model with sigma=16
model.set_calibration(sigma=16.0)
mdr_high_gamma = time_iteration(model, verbose=True)


# In[81]:


# now we compare the decision rules with low and high risk aversion


# In[82]:


tab0 = tabulate(model, mdr, 'l', i0=0)
tab1 = tabulate(model, mdr, 'l', i0=1)
tab0_hg = tabulate(model, mdr_high_gamma, 'l', i0=0)
tab1_hg = tabulate(model, mdr_high_gamma, 'l', i0=1)


# In[83]:


plt.plot(tab0['l'], tab0['b'], label='$b_t$ (bad)' )
plt.plot(tab0['l'], tab1['b'],  label='$b_t$ (good)' )

plt.plot(tab0['l'], tab0_hg['b'], label='$b_t$ (bad) [high gamma]' )
plt.plot(tab0['l'], tab1_hg['b'], label='$b_t$ (good) [high gamma]' )
plt.plot(tab0['l'], tab0['l'], linestyle='--', color='black', label='')
plt.plot(tab0['l'], -0.2*tab0['c'], linestyle='--', color='black', label='')
plt.legend(loc= 'upper left')
plt.grid()

