#!/usr/bin/env python
# coding: utf-8

# # 第7章：用布林陣列篩選特定的資料

# ## 7.1 計算布林陣列的統計資訊

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie[['duration']].head(10)


# In[2]:


movie_2_hours = movie['duration'] > 120
movie_2_hours.head(10)


# In[3]:


movie_2_hours.sum()


# In[4]:


movie_2_hours.mean()*100


# In[5]:


movie['duration'].dropna().gt(120).mean()*100


# In[6]:


movie_2_hours.describe()


# In[7]:


movie_2_hours.astype(int).describe()


# In[8]:


movie['duration'].dropna().gt(120).value_counts(normalize=True)


# In[9]:


actors = movie[['actor_1_facebook_likes',
                'actor_2_facebook_likes']].dropna()
(actors['actor_1_facebook_likes'] > actors['actor_2_facebook_likes']).mean()


# ## 7.2 設定多個布林條件

# In[10]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')


# In[11]:


criteria1 = movie.imdb_score > 8
criteria2 = movie.content_rating == 'PG-13'
criteria3 = ((movie.title_year < 2000) | (movie.title_year > 2009))


# In[12]:


criteria_final = criteria1 & criteria2 & criteria3
criteria_final.head()


# In[13]:


5 < 10 and 3 > 4


# In[14]:


# movie.title_year < 2000 | movie.title_year > 2009


# ## 7.3  以布林陣列來進行過濾

# In[15]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
crit_a1 = movie.imdb_score > 8
crit_a2 = movie.content_rating == 'PG-13'
crit_a3 = (movie.title_year < 2000) | (movie.title_year > 2009)
final_crit_a = crit_a1 & crit_a2 & crit_a3


# In[16]:


crit_b1 = movie.imdb_score < 5
crit_b2 = movie.content_rating == 'R'
crit_b3 = ((movie.title_year >= 2000) & (movie.title_year <= 2010))
final_crit_b = crit_b1 & crit_b2 & crit_b3


# In[17]:


final_crit_all = final_crit_a | final_crit_b
final_crit_all.head()


# In[18]:


movie[final_crit_all].head()


# In[19]:


movie.loc[final_crit_all].head()


# In[20]:


cols = ['imdb_score', 'content_rating', 'title_year']
movie_filtered = movie.loc[final_crit_all, cols]
movie_filtered.head(10)


# In[21]:


# movie.iloc[final_crit_all]


# In[22]:


movie.iloc[final_crit_all.to_numpy()]


# ## 布林選取 vs 索引選取

# In[23]:


college = pd.read_csv('data/college.csv')
college[college['STABBR'] == 'TX'].head()


# In[24]:


college2 = college.set_index('STABBR')
college2.loc['TX'].head()


# In[25]:


get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")


# In[26]:


get_ipython().run_line_magic('timeit', "college2.loc['TX']")


# In[27]:


get_ipython().run_line_magic('timeit', "college2 = college.set_index('STABBR')")


# In[28]:


states = ['TX', 'CA', 'NY']
college[college['STABBR'].isin(states)]


# In[29]:


college2.loc[states]


# ## 7.5 用唯一或已排序的索引標籤來選取資料

# In[30]:


college = pd.read_csv('data/college.csv')
college2 = college.set_index('STABBR')
college2.index.is_monotonic


# In[31]:


college3 = college2.sort_index()
college3.index.is_monotonic


# In[32]:


get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")


# In[33]:


get_ipython().run_line_magic('timeit', "college2.loc['TX']")


# In[34]:


get_ipython().run_line_magic('timeit', "college3.loc['TX']")


# In[35]:


college_unique = college.set_index('INSTNM')
college_unique.index.is_unique


# In[36]:


college[college['INSTNM'] == 'Stanford University']


# In[37]:


college_unique.loc['Stanford University']


# In[38]:


college_unique.loc[['Stanford University']]


# In[39]:


get_ipython().run_line_magic('timeit', "college[college['INSTNM'] == 'Stanford University']")


# In[40]:


get_ipython().run_line_magic('timeit', "college_unique.loc[['Stanford University']]")


# In[41]:


college.index = college['CITY'] + ', ' + college['STABBR']
college = college.sort_index()
college.head()


# In[42]:


college.loc['Miami, FL'].head()


# In[43]:


get_ipython().run_cell_magic('timeit', '', "crit1 = college['CITY'] == 'Miami'\ncrit2 = college['STABBR'] == 'FL'\ncollege[crit1 & crit2]")


# In[44]:


get_ipython().run_line_magic('timeit', "college.loc['Miami, FL']")


# ## 7.6 利用Pandas實現SQL中的功能

# In[45]:


employee = pd.read_csv('data/employee.csv')


# In[46]:


employee.dtypes


# In[47]:


employee.DEPARTMENT.value_counts().head()


# In[48]:


employee.GENDER.value_counts()


# In[49]:


employee.BASE_SALARY.describe()


# In[50]:


depts = ['Houston Police Department-HPD', 'Houston Fire Department (HFD)']
criteria_dept = employee.DEPARTMENT.isin(depts)
criteria_gender = employee.GENDER == 'Female'
criteria_sal = ((employee.BASE_SALARY >= 80000) & (employee.BASE_SALARY <= 120000))


# In[51]:


criteria_final = (criteria_dept &
                  criteria_gender &
                  criteria_sal)


# In[52]:


select_columns = ['UNIQUE_ID', 'DEPARTMENT',
                  'GENDER', 'BASE_SALARY']
employee.loc[criteria_final, select_columns].head()


# In[53]:


criteria_sal = employee.BASE_SALARY.between(80_000, 120_000)


# In[54]:


top_5_depts = employee.DEPARTMENT.value_counts().index[:5]
criteria = ~employee.DEPARTMENT.isin(top_5_depts)
employee[criteria]


# ## 7.7 使用query方法提高布林選取的可讀性

# In[55]:


employee = pd.read_csv('data/employee.csv')
depts = ['Houston Police Department-HPD',
         'Houston Fire Department (HFD)']
select_columns = ['UNIQUE_ID', 'DEPARTMENT',
                  'GENDER', 'BASE_SALARY']


# In[56]:


qs = "DEPARTMENT in @depts "     " and GENDER == 'Female' "     " and 80000 <= BASE_SALARY <= 120000"
emp_filtered = employee.query(qs)
emp_filtered[select_columns].head()


# In[57]:


top10_depts = (employee.DEPARTMENT.value_counts() 
               .index[:10].tolist()
              )
qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
employee_filtered2 = employee.query(qs)
employee_filtered2.head()


# ## 7.8 使用where()維持Series的大小

# In[58]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
fb_likes = movie['actor_1_facebook_likes'].dropna()
fb_likes.head()


# In[59]:


fb_likes.describe()


# In[60]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fb_likes.hist(ax=ax)


# In[61]:


criteria_high = fb_likes < 20_000
criteria_high.mean()*100


# In[62]:


fb_likes.where(criteria_high).head()


# In[63]:


fb_likes.where(criteria_high, other=20000).head()


# In[64]:


criteria_low = fb_likes > 300
fb_likes_cap = (fb_likes
                .where(criteria_high, other=20_000)
                .where(criteria_low, 300)
               ) 
fb_likes_cap.head()


# In[65]:


len(fb_likes), len(fb_likes_cap)


# In[66]:


fig, ax = plt.subplots(figsize=(10, 8))
fb_likes_cap.hist(ax=ax)


# In[67]:


fb_likes_cap2 = fb_likes.clip(lower=300, upper=20000)
fb_likes_cap2.equals(fb_likes_cap)


# ## 7.9 對DataFrame的列進行遮罩

# In[68]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['title_year'] >= 2010
c2 = movie['title_year'].isna()
criteria = c1 | c2


# In[69]:


movie.mask(criteria).head()


# In[70]:


movie_mask = (movie.mask(criteria).dropna(how='all'))
movie_mask.head()


# In[71]:


movie_boolean = movie[movie['title_year'] < 2010]
movie_mask.equals(movie_boolean)


# In[72]:


movie_mask.shape == movie_boolean.shape


# In[73]:


movie_mask.dtypes == movie_boolean.dtypes


# In[74]:


from pandas.testing import assert_frame_equal
assert_frame_equal(movie_boolean, movie_mask,
                   check_dtype=False)


# In[75]:


get_ipython().run_line_magic('timeit', "movie.mask(criteria).dropna(how='all')")


# In[76]:


get_ipython().run_line_magic('timeit', "movie[movie['title_year'] < 2010]")


# ## 7.10 以布林陣列、位置數字和標籤選擇資料

# In[77]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['content_rating'] == 'G'
c2 = movie['imdb_score'] < 4
criteria = c1 & c2


# In[78]:


movie_loc = movie.loc[criteria]
movie_loc.head()


# In[79]:


movie_loc.equals(movie[criteria])


# In[80]:


# movie_iloc = movie.iloc[criteria]


# In[81]:


movie_iloc = movie.iloc[criteria.to_numpy()]
movie_iloc.equals(movie_loc)


# In[82]:


criteria_col = movie.dtypes == np.int64
criteria_col.head()


# In[83]:


movie.loc[:, criteria_col].head()


# In[84]:


movie.iloc[:, criteria_col.to_numpy()].head()


# In[85]:


cols = ['content_rating', 'imdb_score', 'title_year', 'gross']
movie.loc[criteria, cols].sort_values('imdb_score')


# In[86]:


col_index = [movie.columns.get_loc(col) for col in cols]
col_index


# In[87]:


movie.iloc[criteria.to_numpy(), col_index].sort_values('imdb_score')


# In[88]:


a = criteria.to_numpy()
a[:5]


# In[89]:


len(a), len(criteria)


# In[90]:


movie.select_dtypes(int)

