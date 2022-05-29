#!/usr/bin/env python
# coding: utf-8

# # Fundamentals

# In[2]:


import numpy as np


# ## create array

# ### np.arange

# * 我想建立像 `range()` 這種版本的 array

# In[6]:


np.arange(10) # 0~9


# * 我想用間隔來建立 array

# In[4]:


np.arange(0, 10, 2) # start=0, stop = 10(所以不包含10), 間距=2


# ### np.linspace

# * 我想用長度來建立 array

# In[7]:


np.linspace(0, 10, 5) # start=0, end = 10(所以包含10), 均分成 5 個資料點


# In[ ]:




