#!/usr/bin/env python
# coding: utf-8

# In[31]:


from numpy import *
from matplotlib import pyplot as plt


# In[32]:


from dolo import *


# In[37]:


filename =  '../models/rbc_taxes.yaml'


# In[38]:


model = yaml_import(filename)


# The model defined in `rbc_taxes.yaml` is the `rbc` model, with an agregate tax `g` that is proportional to income. 

# In[39]:


model.calibration


# In[40]:


model.residuals()


# We want to compute the adjustment of the economy when this tax, goes back progressively from 10% to 0%, over 10 periods.

# In[41]:


exo_g = linspace(0.1,0,10) # this is a vector of size 10
exo_g = atleast_2d(exo_g).T # the solver expects a 1x10 vector
print(exo_g.shape)


# In[42]:


exo_g


# In[44]:


# Let's solve for the optimal adjustment by assuming that the
# economy returns to steady-state after T=50 periods.
from dolo.algos.perfect_foresight import deterministic_solve
sim = deterministic_solve(model, shocks=exo_g, T=50)
display(sim) # it returns a timeseries object


# In[45]:


model


# In[47]:


plt.plot(figsize=(10,10))
plt.subplot(221)
plt.plot(sim['k'], label='capital')
plt.plot(sim['y'], label='production')
plt.legend()
plt.subplot(222)
plt.plot(sim['g'], label='gvt. spending')
plt.plot(sim['c'], label='consumption')
plt.legend()
plt.subplot(223)
plt.plot(sim['n'], label='work')
plt.plot(sim['i'], label='investment')
plt.legend()
plt.subplot(224)
plt.plot(sim['w'], label='wages')
plt.legend()


# In[ ]:




