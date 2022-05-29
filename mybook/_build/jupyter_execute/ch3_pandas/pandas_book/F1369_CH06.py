#!/usr/bin/env python
# coding: utf-8

# # 第6章：選取資料的子集

# ## 6.1 選取一筆或多筆Series資料

# In[55]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)

college = pd.read_csv('data/college.csv', index_col='INSTNM')
city = college['CITY']
city


# In[56]:


city['Alabama A & M University']


# In[57]:


city.loc['Alabama A & M University']


# In[58]:


city.iloc[0]


# In[59]:


city[['Alabama A & M University', 'Alabama State University']]


# In[60]:


city.loc[['Alabama A & M University', 'Alabama State University']]


# In[61]:


city.iloc[[0, 4]]


# In[62]:


city['Alabama A & M University': 'Alabama State University']


# In[63]:


city[0:5]


# In[64]:


city.loc['Alabama A & M University': 'Alabama State University']


# In[65]:


city.iloc[0:5]


# In[66]:


alabama_mask = city.isin(['Birmingham', 'Montgomery'])
alabama_mask


# In[67]:


city[alabama_mask]


# In[68]:


s = pd.Series([10, 20, 35, 28], index=[5,2,3,1])
s


# In[69]:


s[0:4]


# In[70]:


s[5]


# In[71]:


s[1]


# In[72]:


college.loc['Alabama A & M University', 'CITY']


# In[73]:


college.iloc[0, 0]


# In[74]:


college.loc[['Alabama A & M University', 'Alabama State University'], 'CITY']


# In[75]:


college.iloc[[0, 4], 0]


# In[76]:


college.loc['Alabama A & M University': 'Alabama State University', 'CITY']


# In[77]:


college.iloc[0:5, 0]


# In[78]:


city.loc['Reid State Technical College':
         'Alabama State University']


# ## 6.2 選取DataFrame的列

# In[79]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.head()


# In[80]:


college.iloc[0]


# In[81]:


college.loc['Alabama A & M University']


# In[82]:


college.iloc[[60, 99, 3]]


# In[83]:


labels = ['University of Alaska Anchorage',
          'International Academy of Hair Design',
          'University of Alabama in Huntsville']
college.loc[labels]


# In[84]:


college.iloc[99:102]


# In[85]:


start = 'International Academy of Hair Design'
stop = 'Mesa Community College'
college.loc[start:stop]


# ## 6.3 同時選取DataFrame的列與欄位

# In[86]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.iloc[:3, :4]


# In[87]:


college.loc[:'Amridge University', :'MENONLY']


# In[88]:


college.iloc[:, [4,6]].head() 


# In[89]:


college.loc[:, ['WOMENONLY', 'SATVRMID']].head()


# In[90]:


college.iloc[5, -4]


# In[91]:


college.loc['The University of Alabama', 'PCTFLOAN']


# In[92]:


college.iloc[10:20:2, 5]


# In[93]:


start = 'Birmingham Southern College'
stop = 'New Beginning College of Cosmetology'
college.loc[start:stop:2, 'RELAFFIL']


# In[94]:


college.iloc[:10]


# In[95]:


college.iloc[:10, :]


# ## 6.4 混用位置與標籤來選取資料

# In[96]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')


# In[97]:


col_start = college.columns.get_loc('UGDS_WHITE')
col_end = college.columns.get_loc('UGDS_UNKN') + 1
col_start, col_end


# In[98]:


college.iloc[:5, col_start:col_end]


# In[99]:


row_start = college.index[10]
row_end = college.index[15]
row_start, row_end


# In[100]:


college.loc[row_start:row_end, 'UGDS_WHITE':'UGDS_UNKN']


# In[101]:


# college.ix[10:16, 'UGDS_WHITE':'UGDS_UNKN']


# In[102]:


college.iloc[10:16].loc[:, 'UGDS_WHITE':'UGDS_UNKN']


# ## 6.5 按標籤的字母順序進行切片

# In[103]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')


# In[104]:


# college.loc['Sp':'Su']


# In[105]:


college = college.sort_index()


# In[106]:


college.loc['Sp':'Su']


# In[107]:


college = college.sort_index(ascending=False)
college.index.is_monotonic_decreasing


# In[108]:


college.loc['E':'B']

