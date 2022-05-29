#!/usr/bin/env python
# coding: utf-8

# # 第2章：DataFrame的運算技巧

# ## 2.1 選取多個DataFrame的欄位

# In[1]:


import pandas as pd
import numpy as np

movies = pd.read_csv('data/movie.csv')
movie_actor_director = movies[['actor_1_name', 'actor_2_name',
                               'actor_3_name', 'director_name']]
movie_actor_director.head()


# In[2]:


type(movies[['director_name']])


# In[3]:


type(movies['director_name'])


# In[4]:


type(movies.loc[:, ['director_name']])


# In[5]:


type(movies.loc[:, 'director_name'])


# In[6]:


cols = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
movie_actor_director = movies[cols]


# In[7]:


# movies['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']


# ## 2.2 用方法（methods）選取欄位

# In[7]:


movies = pd.read_csv('data/movie.csv')
movies.dtypes.value_counts()


# In[8]:


movies.select_dtypes(include='object').head()


# In[10]:


def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', ''))
movies = movies.rename(columns=shorten)
movies.select_dtypes(include='number').head()


# In[11]:


movies.select_dtypes(include=['int', 'object']).head()


# In[12]:


movies.select_dtypes(exclude='float').head()


# In[13]:


movies.filter(like='fb').head()


# In[14]:


cols = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
movies.filter(items=cols).head()


# In[15]:


movies.filter(regex=r'\d').head()


# ## 2.3 對欄位名稱進行排序

# In[16]:


movies = pd.read_csv('data/movie.csv')
def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', '')
    )
movies = movies.rename(columns=shorten)


# In[17]:


movies.columns


# In[18]:


cat_core = ['movie_title', 'title_year', 'content_rating', 'genres']
cat_people = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
cat_other = ['color', 'country', 'language', 'plot_keywords', 'movie_imdb_link']
cont_fb = ['director_fb', 'actor_1_fb', 'actor_2_fb', 'actor_3_fb',
           'cast_total_fb', 'movie_fb']
cont_finance = ['budget', 'gross']
cont_num_reviews = ['num_voted_users', 'num_user', 'num_critic']
cont_other = ['imdb_score', 'duration', 'aspect_ratio', 'facenumber_in_poster']


# In[19]:


new_col_order = cat_core + cat_people +                 cat_other + cont_fb +                 cont_finance + cont_num_reviews +                 cont_other
set(movies.columns) == set(new_col_order)


# In[20]:


movies[new_col_order].head()


# ## 2.4 DataFrame的統計方法

# In[21]:


movies = pd.read_csv('data/movie.csv')
movies.shape


# In[22]:


movies.size


# In[23]:


movies.ndim


# In[24]:


len(movies)


# In[25]:


movies.count()


# In[26]:


movies.min()


# In[27]:


movies.describe().T


# In[28]:


movies.describe(percentiles=[.99]).T


# In[29]:


movies.min(skipna=False)


# ## 2.5 串連DataFrame的方法

# In[30]:


movies = pd.read_csv('data/movie.csv')
def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', ''))
movies = movies.rename(columns=shorten)
movies.isna().head()


# In[31]:


(movies.isna().sum().head())


# In[32]:


movies.isna().sum().sum()


# In[33]:


movies.isna().any()


# In[34]:


movies.isna().any().any()


# In[35]:


movies[['color']].max()


# In[36]:


movies.select_dtypes(['object']).fillna('')


# In[37]:


(movies.select_dtypes(['object'])
       .fillna('')
       .max())


# ## 2.6 DataFrame的算符運算

# In[38]:


# colleges = pd.read_csv('data/college.csv')
# colleges + 5


# In[39]:


colleges = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = colleges.filter(like='UGDS_')
college_ugds.head()


# In[40]:


name = 'Northwest-Shoals Community College'
college_ugds.loc[name]


# In[41]:


college_ugds.loc[name].round(2)


# In[42]:


(college_ugds.loc[name] + .0001).round(2)


# In[43]:


college_ugds + .00501


# In[44]:


.045+.005


# In[45]:


(college_ugds + .00501) // .01


# In[46]:


college_ugds_op_round = (college_ugds + .00501) // .01 / 100
college_ugds_op_round.head()


# In[47]:


college_ugds_round = (college_ugds + .00001).round(2)
college_ugds_round


# In[48]:


college_ugds_op_round.equals(college_ugds_round)


# In[49]:


college2 = (college_ugds
    .add(.00501) 
    .floordiv(.01) 
    .div(100)
)
college2.equals(college_ugds_op_round)


# ## 2.7 比較缺失值

# In[50]:


np.nan == np.nan


# In[51]:


None == None


# In[52]:


np.nan > 5


# In[53]:


5 > np.nan


# In[54]:


np.nan != 5


# In[55]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')


# In[56]:


college_ugds == .0019


# In[57]:


college_self_compare = college_ugds == college_ugds
college_self_compare.head()


# In[58]:


college_self_compare.all()


# In[59]:


(college_ugds == np.nan).sum()


# In[60]:


college_ugds.isna().sum()


# In[61]:


college_ugds.equals(college_ugds)


# In[62]:


college_ugds.eq(.0019)    


# In[63]:


from pandas.testing import assert_frame_equal
assert_frame_equal(college_ugds, college_ugds) is None


# ## 2.8 轉置DataFrame運算的方向

# In[9]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()


# In[10]:


college_ugds.count()


# In[11]:


college_ugds.count(axis='columns').head()


# In[67]:


college_ugds.sum(axis='columns').head()


# In[68]:


college_ugds.median(axis='index')


# In[69]:


college_ugds_cumsum = college_ugds.cumsum(axis=1)
college_ugds_cumsum.head()


# ## 2.9 案例演練：確定大學校園的多樣性

# In[12]:


pd.read_csv('data/college_diversity.csv', index_col='School')


# In[13]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')


# In[14]:


college_ugds


# In[16]:


(college_ugds.isna()
             .sum(axis='columns')
             .sort_values(ascending=False)
             #.head()
)


# In[73]:


college_ugds = college_ugds.dropna(how='all')
college_ugds.isna().sum()


# In[74]:


college_ugds.ge(0.15)


# In[75]:


diversity_metric = college_ugds.ge(.15).sum(axis='columns')
diversity_metric.head()


# In[76]:


diversity_metric.value_counts()


# In[77]:


diversity_metric.sort_values(ascending=False).head()


# In[78]:


college_ugds.loc[['Regency Beauty Institute-Austin',
                  'Central Texas Beauty College-Temple']]


# In[79]:


us_news_top =['Rutgers University-Newark',
              'Andrews University',
              'Stanford University',
              'University of Houston',
              'University of Nevada-Las Vegas']
diversity_metric.loc[us_news_top]


# In[80]:


(college_ugds
   .max(axis=1)
   .sort_values(ascending=False)
   .head(10)
)


# In[81]:


(college_ugds > .01).all(axis=1).any()

