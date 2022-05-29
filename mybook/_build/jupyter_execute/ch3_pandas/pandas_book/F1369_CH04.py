#!/usr/bin/env python
# coding: utf-8

# # 第4章：開始資料分析

# In[2]:


import pandas as pd
import numpy as np
# pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## 讀取資料 & overview

# * 先讀資料，看一下長相：

# In[4]:


college = pd.read_csv('data/college.csv')
college.head()


# * 看一下 shape

# In[5]:


college.shape


# * 知道有 7535 筆資料，然後欄位數是 27
# * 看一下各欄位的 data type

# In[6]:


college.info()


# * 大概統計一下各種 data type 的個數：

# In[11]:


college.dtypes.value_counts()


# * 可以看到，幾乎都是連續型 (flat64 與 int64 共 22 個), 類別型有 5 個
# * 來看一下連續型的 summary

# In[12]:


college.describe().T


# * 看一下類別型的 summary

# In[14]:


college.describe(include=[object]).T


# In[16]:


college.describe(include=[object, np.number]).T


# In[17]:


college.describe(include=[np.number],
                 percentiles=[.01, .05, .10, .25, .5,
                              .75, .9, .95, .99]).T


# ## 資料字典

# In[18]:


pd.read_csv('data/college_data_dictionary.csv')


# ## 改變資料型別以減少記憶體用量

# In[20]:


college = pd.read_csv('data/college.csv')
different_cols = ['RELAFFIL', 'SATMTMID', 'CURROPER', 'INSTNM', 'STABBR'] # 不同資料型別取幾個當代表
col2 = college.loc[:, different_cols]
col2.head()


# In[21]:


col2.dtypes


# In[22]:


# 記憶體使用量
original_mem = col2.memory_usage(deep=True)
original_mem


# * 但 RELAFFIL 只有 0, 1 兩種取值：

# In[23]:


col2['RELAFFIL'].value_counts()


# * 所以把他改成 int8 就足夠了：

# In[24]:


col2['RELAFFIL'] = col2['RELAFFIL'].astype(np.int8)    


# In[25]:


col2.dtypes


# In[26]:


col2.memory_usage(deep=True)


# * RELAFFIL 的記憶體使用量從 60280 減到 7535
# * 再看一下兩個 object 類型的欄位，有多少 unique 的值

# In[27]:


col2.select_dtypes(include=['object']).nunique()


# * 可以看到 STABBR 只有 59 個不同的值，那我就把它改為 category 型別 (只存整數 0, 1, ..., 58)

# In[28]:


col2['STABBR'] = col2['STABBR'].astype('category')
col2.dtypes


# In[29]:


new_mem = col2.memory_usage(deep=True)
new_mem


# In[30]:


new_mem / original_mem


# ## 資料的排序

# In[31]:


movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie2.head()


# * 取 imdb_score 分數最高的前 100 名

# In[33]:


movie2.nlargest(100, 'imdb_score').head()


# * 也可用排序後 slice 的方式

# In[35]:


movie2.sort_values("imdb_score", ascending=False).head(100).head()


# * 如果要從 imdb_score 前 100 名中，找出 budgets 最少的前 5 名，可以這樣做：

# In[36]:


(movie2.nlargest(100, 'imdb_score')
       .nsmallest(5, 'budget')
)


# * 也可用排序的方式做

# In[37]:


(
    movie2
        .sort_values("imdb_score", ascending=False)
        .head(100)
        .sort_values("budget")
        .head(5)
)

