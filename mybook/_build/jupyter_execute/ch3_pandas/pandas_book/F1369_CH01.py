#!/usr/bin/env python
# coding: utf-8

# # 第1章: DataFrame及Series的基本操作

# In[ ]:


import pandas as pd
import numpy as np
# pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Series的相關操作

# ### Series 有向量運算的特性 (可作 vectorize 運算, 有 broadcasting)

# In[3]:


movies = pd.read_csv('data/movie.csv')
imdb_score = movies['imdb_score']
imdb_score


# In[6]:


imdb_score + 1 # broadcasting


# In[ ]:


imdb_score // 7


# In[48]:


imdb_score > 7


# In[8]:


director = movies['director_name']
director == 'James Cameron'


# ### 把 運算符 改成 method, 可用方法

# * 數值運算：
#   * `+` -> `.add`. 
#   * `-` -> `.sub`
#   * `*` -> `.mul`. 
#   * `/` -> `.div`. 
#   * `//` -> `.floordiv`. 
#   * `%` -> `.mod`. 
#   * `**` -> `.pow`  
# * 比較運算:
#   * `<` -> `.lt`. 
#   * `<=` -> `.le`. 
#   * `>` -> `.gt`. 
#   * `>=` -> `.ge`. 
#   * `==` -> `.eq`  
#   * `!=` -> `.ne`

# In[9]:


money = pd.Series([100, 20, None])
money - 15


# In[10]:


money.sub(15)


# In[11]:


money.sub(15, fill_value = 0) # 先把 na 補 0 ，再計算


# In[12]:


money.gt(10)


# ## 1.4 串連Series的方法

# In[15]:


movies = pd.read_csv('data/movie.csv')
fb_likes = movies['actor_1_facebook_likes']
director = movies['director_name']


# ### 可以直接一路`.`下去

# In[16]:


director.value_counts().head(3)


# In[17]:


fb_likes.isna().sum()


# ### 也可用括號括起來，就可以用 enter 來換行 (可註解)

# In[58]:


(fb_likes.fillna(0)
         .astype(int)
         .head()
)


# In[20]:


(fb_likes.fillna(0)
         #.astype(int)
         .head()
)


# ### 也可用 `\` 來分隔 (不可註解)

# In[19]:


fb_likes     .fillna(0)     .astype(int)     .head()


# In[21]:


fb_likes     .fillna(0)     #.astype(int) \
    .head()


# ## 更改欄位名稱

# In[22]:


movies = pd.read_csv('data/movie.csv')


# ### 用 `rename` 來改

# In[23]:


col_map = {'director_name':'Director Name'} 


# In[24]:


movies.rename(columns=col_map).head()


# ### 清理不乾淨的欄位名稱

# * 例如，我想把現在的欄位名稱：  
#   * 前後的空白都去掉
#   * 轉小寫
#   * `director.name` 的 `.` 改成 `_` -> `director_name`. 
# * 那可以這樣做

# In[28]:


aa = "wHaT.ever  "
aa.strip().lower().replace('.', '_')


# * 就把上面的作法套到欄位名稱上就好：

# In[29]:


cols = [col.strip().lower().replace('.', '_') for col in movies.columns]
movies.columns = cols
movies.head(3)


# ## 刪除column

# In[43]:


movies = pd.read_csv('data/movie.csv')
movies.columns


# * 刪掉第一個 column

# In[44]:


movies = movies.drop('color', axis = 1)
movies.columns


# * 刪掉前五個 column

# In[45]:


del_columns = ['director_name', 'num_critic_for_reviews', 'duration']
movies = movies.drop(del_columns, axis = 1)
movies.columns


# ## 新增 column

# In[46]:


movies = pd.read_csv('data/movie.csv')
target_columns = ['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']
movies = movies[target_columns]
movies


# ### 直接用 `[]` 來處理

# * 新增一個常數

# In[47]:


movies['has_seen'] = 0
movies.head()


# * 新增對各欄位做向量運算的結果 (加總觀眾對 3 個演員 + 1 個導演 的讚數)

# In[48]:


# 加總法 1: 若欄位內有 NA，加總完會有 NA
total1 = (movies['actor_1_facebook_likes'] +
         movies['actor_2_facebook_likes'] + 
         movies['actor_3_facebook_likes'] + 
         movies['director_facebook_likes'])

# 加總法 2: 用 dataframe 的 sum method，加總時就會把 NA 先補 0 再相加
cols = ['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']
total2 = movies[cols].sum(axis='columns')

# 新增欄位
movies["total1"] = total1
movies["total2"] = total2


# In[49]:


movies.info()


# * 可以看到 total1 和 total2 的 Non-Null Count 有差別

# ### 用 `.assign` method

# In[53]:


movies = pd.read_csv('data/movie.csv')
target_columns = ['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']
movies = movies[target_columns]
movies


# #### 先算好再 assign 進去

# * 作法和剛剛都一樣，都是先算出結果，再 assign 到新欄位

# In[54]:


# 加總法 1: 若欄位內有 NA，加總完會有 NA
total1 = (movies['actor_1_facebook_likes'] +
         movies['actor_2_facebook_likes'] + 
         movies['actor_3_facebook_likes'] + 
         movies['director_facebook_likes'])

# 加總法 2: 用 dataframe 的 sum method，加總時就會把 NA 先補 0 再相加
cols = ['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']
total2 = movies[cols].sum(axis='columns')

# 新增欄位
movies = movies.assign(total1 = total1,
                      total2 = total2)
movies


# * 從這個角度來看，已經很像 R 的 mutate 做法了，可以這樣寫更像：

# In[56]:


# 原始資料
movies = pd.read_csv('data/movie.csv')
target_columns = ['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']
movies = movies[target_columns]

# 開始做 mutate
movies = movies.assign(
    total1 = (movies['actor_1_facebook_likes'] +
         movies['actor_2_facebook_likes'] + 
         movies['actor_3_facebook_likes'] + 
         movies['director_facebook_likes']),
    total2 = movies[['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']].sum(axis='columns')
)

movies


# * 的確跟 R 很像了，但很討厭的就是，我還是要一直寫 dataframe 的名稱 `movies`  
# * 而且，如果想像 R 一路做 pipe 下去的話，那就 GG 了

# In[58]:


movies = pd.read_csv('data/movie.csv')
target_columns = ['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']
movies = movies[target_columns]
movies


# In[66]:


# 原始資料
movies = pd.read_csv('data/movie.csv')
target_columns = ['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']
movies = movies[target_columns]

# 開始做 mutate
movies = (
    movies
        .query("director_facebook_likes > 500") # 先篩選
        .sort_values("director_facebook_likes")
        .assign(total1 = 
                   (movies['actor_1_facebook_likes'] +
                    movies['actor_2_facebook_likes'] + 
                    movies['actor_3_facebook_likes'] + 
                    movies['director_facebook_likes'])
               )
)

movies


# In[65]:


# 原始資料
movies = pd.read_csv('data/movie.csv')
target_columns = ['actor_1_facebook_likes','actor_2_facebook_likes',
        'actor_3_facebook_likes','director_facebook_likes']
movies = movies[target_columns]

# 開始做 mutate
movies = (
    movies
        .query("director_facebook_likes > 500") # 先篩選
        .sort_values("director_facebook_likes")
        .assign(total1 = lambda x:
                   (x['actor_1_facebook_likes'] +
                    x['actor_2_facebook_likes'] + 
                    x['actor_3_facebook_likes'] + 
                    x['director_facebook_likes'])
               )
)

movies


# In[79]:


def sum_likes(df):
    return df[[c for c in df.columns
               if 'like' in c]].sum(axis=1)
movies.assign(total_likes=sum_likes).head(5)


# In[82]:


def cast_like_gt_actor_director(df):
    return df['cast_total_facebook_likes'] >= df['total_likes']

df2 = (movies.assign(total_likes=total,
                     is_cast_likes_more = cast_like_gt_actor_director)
      )


# In[83]:


df2['is_cast_likes_more'].all()


# In[84]:


df2 = df2.drop(columns='total_likes')


# In[85]:


actor_sum = (movies[[c for c in movies.columns if 'actor_' in c and '_likes' in c]]
             .sum(axis='columns'))

actor_sum.head(5)


# In[86]:


movies['cast_total_facebook_likes'] >= actor_sum


# In[87]:


movies['cast_total_facebook_likes'].ge(actor_sum)


# In[88]:


movies['cast_total_facebook_likes'].ge(actor_sum).all()


# In[89]:


pct_like = (actor_sum
            .div(movies['cast_total_facebook_likes'])
)


# In[90]:


pct_like.describe()


# In[91]:


pd.Series(pct_like.values,
          index=movies['movie_title'].values).head()


# In[92]:


profit_index = movies.columns.get_loc('gross') + 1
profit_index


# In[93]:


movies.insert(loc=profit_index,
              column='profit',
              value=movies['gross'] - movies['budget'])


# In[94]:


del movies['director_name']

