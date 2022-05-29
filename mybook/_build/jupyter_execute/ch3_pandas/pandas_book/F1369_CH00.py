#!/usr/bin/env python
# coding: utf-8

# # 第0章：Pandas套件的基礎

# ## 0.1 DataFrame物件

# In[2]:


import pandas as pd
import numpy as np
# pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# In[3]:


movies = pd.read_csv('data/movie.csv')
movies


# In[26]:


movies.head()


# In[27]:


movies.head(3)


# ## DataFrame的屬性（attributes）

# * 一張 pd.DataFrame 由 index, columns 和 data 所組成：
#   * .index: 就是 row name，為 numpy 的 第 0 軸。 type 是 pandas 的 index 物件
#   * .columns: 就是 column name，為 numpy 的 第1軸，type是 pandas 的 index 物件
#   * .values: 就是存放內容的 matrix，為 numpy 的 ndarray 物件

# In[16]:


movies = pd.read_csv('data/movie.csv')


# In[17]:


print(movies.index)
print(type(movies.index))


# In[18]:


print(movies.columns)
print(type(movies.columns))


# In[20]:


print(movies.values)
print(type(movies.values))


# * 如果想像 R 來取 rowname, colname，那記得把他轉回 numpy 格式比較好用：

# In[14]:


# 取 row name
movies.index.to_numpy()


# In[15]:


# 取 col name
movies.columns.to_numpy()


# ## Series物件

# In[40]:


fruit = pd.Series(['apple', 'banana', 'grape', 'pineapple'], index=['a', 'b', 'c', 'd'])
fruit


# In[41]:


fruit = pd.Series(['apple', 'banana', 'grape', 'pineapple'])
fruit


# In[42]:


fruit.dtypes


# In[43]:


fruit.size


# ## 0.4 Pandas中的資料型別

# In[44]:


movies = pd.read_csv('data/movie.csv')
movies.dtypes


# In[45]:


movies.dtypes.value_counts()


# In[46]:


movies.info()

