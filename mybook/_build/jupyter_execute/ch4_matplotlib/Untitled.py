#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd


# * fig 是 container which hold everything you see on the page
# * ax 是 part of the page that holds data, it is the canvas
# * adding data to axes

# In[5]:


seattle_weather = pd.read_csv("data/seattle_weather.csv")
austin_weather = pd.read_csv("data/austin_weather.csv")


# In[6]:


seattle_weather


# In[18]:


# Import the matplotlib.pyplot submodule and name it plt
import matplotlib.pyplot as plt

# Create a Figure and an Axes with plt.subplots
fig, ax = plt.subplots()

# Plot MLY-PRCP-NORMAL from seattle_weather against the MONTH
ax.plot(seattle_weather["DATE"], 
        seattle_weather["MLY-PRCP-NORMAL"],
       color = "b", # blue
       marker = "o", # 點的形狀
       linestyle = "--" # 線的形狀
       )

# Plot MLY-PRCP-NORMAL from austin_weather against MONTH
ax.plot(austin_weather["DATE"], 
        austin_weather["MLY-PRCP-NORMAL"],
        color = "r", # 紅色
       marker = "v", # 點的形狀
       linestyle = "--" # 線的形狀
       )
# Customize the x-axis label
ax.set_xlabel("Time (months)")

# Customize the y-axis label
ax.set_ylabel("Precipitation (inches)")

# Add the title
ax.set_title("Weather patterns in Austin and Seattle")
# Call the show function
plt.show()


# In[3]:


fig, ax = plt.subplots(3, 2)


# In[4]:


fig, ax = plt.subplots(2,1)


# In[ ]:




