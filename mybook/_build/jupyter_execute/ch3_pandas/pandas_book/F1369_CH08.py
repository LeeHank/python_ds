#!/usr/bin/env python
# coding: utf-8

# # 第8章：索引對齊與尋找欄位最大值

# ## 8.1 檢驗Index物件

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)

college = pd.read_csv('data/college.csv')
columns = college.columns
columns


# In[2]:


columns.to_numpy()


# In[3]:


columns[5]


# In[4]:


columns[[1,8,10]]


# In[5]:


columns[-7:-4]


# In[6]:


columns.isna().sum()


# In[7]:


columns + '_A'


# In[8]:


columns > 'G'


# In[9]:


# columns[1] = 'city'


# In[10]:


c1 = columns[:4]
c1


# In[11]:


c2 = columns[2:6]
c2


# In[12]:


c1.union(c2) 


# ## 8.2 笛卡爾積

# In[13]:


s1 = pd.Series(index=list('aaab'), data=np.arange(4))
s1


# In[14]:


s2 = pd.Series(index=list('aabbbc'), data=np.arange(6))
s2


# In[15]:


s1 + s2


# In[16]:


s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('aaabb'), data=np.arange(5))
s1


# In[17]:


s2


# In[18]:


s1+s2


# In[19]:


s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('bbaaa'), data=np.arange(5))
s1 


# In[20]:


s2


# In[21]:


s1+s2


# In[22]:


s3 = pd.Series(index=list('ab'), data=np.arange(2))
s4 = pd.Series(index=list('ba'), data=np.arange(2))
s3 + s4


# ## 8.3 索引爆炸

# In[23]:


employee = pd.read_csv('data/employee.csv', index_col='RACE')
employee.head()


# In[24]:


salary1 = employee['BASE_SALARY']
salary2 = employee['BASE_SALARY']
salary1 is salary2


# In[25]:


salary2 = employee['BASE_SALARY'].copy()
salary1 is salary2


# In[26]:


salary1 = salary1.sort_index()
salary1.head()


# In[27]:


salary2.head()


# In[28]:


salary_add = salary1 + salary2


# In[29]:


salary_add.head()


# In[30]:


len(salary1.index), len(salary2.index), len(salary_add.index)


# In[31]:


index_vc = salary1.index.value_counts(dropna=False)
index_vc


# In[32]:


index_vc.pow(2).sum()


# ## 8.4 填補缺失值

# In[33]:


baseball_14 = pd.read_csv('data/baseball14.csv', index_col='playerID')
baseball_15 = pd.read_csv('data/baseball15.csv', index_col='playerID')
baseball_16 = pd.read_csv('data/baseball16.csv', index_col='playerID')
baseball_14.head()


# In[34]:


baseball_14.index.difference(baseball_15.index)


# In[35]:


baseball_15.index.difference(baseball_14.index)


# In[36]:


hits_14 = baseball_14['H']
hits_15 = baseball_15['H']
hits_16 = baseball_16['H']
hits_14.head()


# In[37]:


hits_15.head()


# In[38]:


hits_16.head()


# In[39]:


(hits_14 + hits_15).head()


# In[40]:


hits_14.add(hits_15, fill_value=0).head()


# In[41]:


hits_total = (hits_14.add(hits_15, fill_value=0)
                     .add(hits_16, fill_value=0))

hits_total.head()


# In[42]:


hits_total.hasnans


# In[43]:


s = pd.Series(index=['a', 'b', 'c', 'd'],
              data=[np.nan, 3, np.nan, 1])
s


# In[44]:


s1 = pd.Series(index=['a', 'b', 'c'],
               data=[np.nan, 6, 10])
s1


# In[45]:


s.add(s1, fill_value=5)


# In[46]:


df_14 = baseball_14[['G','AB', 'R', 'H']]
df_14.head()


# In[47]:


df_15 = baseball_15[['AB', 'R', 'H', 'HR']]
df_15.head()


# In[48]:


(df_14 + df_15).head(10).style.highlight_null('lightgrey')


# In[49]:


(df_14.add(df_15, fill_value=0)
      .head(10)
      .style.highlight_null('lightgrey')
)


# ## 8.5 從不同的DataFrame增加欄位

# In[50]:


employee = pd.read_csv('data/employee.csv')
dept_sal = employee[['DEPARTMENT', 'BASE_SALARY']]
dept_sal


# In[51]:


dept_sal = dept_sal.sort_values(['DEPARTMENT', 'BASE_SALARY'],
                                 ascending=[True, False])
dept_sal


# In[52]:


max_dept_sal = dept_sal.drop_duplicates(subset='DEPARTMENT')
max_dept_sal.head()


# In[53]:


max_dept_sal = max_dept_sal.set_index('DEPARTMENT')
employee = employee.set_index('DEPARTMENT')


# In[54]:


employee = employee.assign(MAX_DEPT_SALARY=max_dept_sal['BASE_SALARY'])
employee


# In[55]:


employee.query('BASE_SALARY > MAX_DEPT_SALARY')


# In[56]:


employee = pd.read_csv('data/employee.csv')
max_dept_sal = (employee[['DEPARTMENT', 'BASE_SALARY']]
                   .sort_values(['DEPARTMENT', 'BASE_SALARY'], 
                                ascending=[True, False])
                   .drop_duplicates(subset='DEPARTMENT')
                   .set_index('DEPARTMENT')
)


# In[57]:


(employee.set_index('DEPARTMENT')
         .assign(MAX_DEPT_SALARY=max_dept_sal['BASE_SALARY']))


# In[58]:


random_salary = (dept_sal.sample(n=10, random_state=42)
                         .set_index('DEPARTMENT'))
random_salary


# In[59]:


# employee['RANDOM_SALARY'] = random_salary['BASE_SALARY']


# In[60]:


max_dept_sal['BASE_SALARY'].head(3)


# In[61]:


(employee.set_index('DEPARTMENT')
         .assign(MAX_SALARY2=max_dept_sal['BASE_SALARY'].head(3))
         .MAX_SALARY2
         .value_counts(dropna=False)
)


# In[62]:


max_sal = (employee.groupby('DEPARTMENT')
                   .BASE_SALARY
                   .transform('max')
)


# In[63]:


employee.assign(MAX_DEPT_SALARY=max_sal)


# ## 8.6 凸顯每一欄位的最大值

# In[64]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.head()


# In[65]:


college.dtypes


# In[66]:


college.MD_EARN_WNE_P10.sample(10, random_state=42)


# In[67]:


college.GRAD_DEBT_MDN_SUPP.sample(10, random_state=42)


# In[68]:


college.MD_EARN_WNE_P10.value_counts()


# In[69]:


college.GRAD_DEBT_MDN_SUPP.value_counts()


# In[70]:


cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')
college.GRAD_DEBT_MDN_SUPP.sample(10, random_state=42)


# In[71]:


college.dtypes.loc[cols]


# In[72]:


college_n = college.select_dtypes('number')
college_n.head()


# In[73]:


binary_only = college_n.nunique() == 2
binary_only.head()


# In[74]:


binary_cols = binary_only[binary_only].index.tolist()
binary_cols


# In[75]:


college_n2 = college_n.drop(columns=binary_cols)
college_n2.head()


# In[76]:


max_cols = college_n2.idxmax()
max_cols


# In[77]:


unique_max_cols = max_cols.unique()
unique_max_cols[:5]


# In[78]:


college_n2.loc[unique_max_cols].style.highlight_max(color='lightgrey')


# In[79]:


def remove_binary_cols(df):
    binary_only = df.nunique() == 2
    cols = binary_only[binary_only].index.tolist()
    return df.drop(columns=cols)

def select_rows_with_max_cols(df):
    max_cols = df.idxmax()
    unique = max_cols.unique()
    return df.loc[unique]

(college
   .assign(
       MD_EARN_WNE_P10=pd.to_numeric(college.MD_EARN_WNE_P10, errors='coerce'),
       GRAD_DEBT_MDN_SUPP=pd.to_numeric(college.GRAD_DEBT_MDN_SUPP, errors='coerce'))
   .select_dtypes('number')
   .pipe(remove_binary_cols)
   .pipe(select_rows_with_max_cols)
   .style.highlight_max(color='lightgrey')
)


# In[80]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_').head()
college_ugds.style.highlight_max(axis='columns',color='lightgrey')


# ## 8.7 串連方法來實現idxmax()的功能

# In[81]:


def remove_binary_cols(df):
    binary_only = df.nunique() == 2
    cols = binary_only[binary_only].index.tolist()
    return df.drop(columns=cols)

college_n = (
    college
    .assign(
        MD_EARN_WNE_P10=pd.to_numeric(
            college.MD_EARN_WNE_P10, errors='coerce'),
        GRAD_DEBT_MDN_SUPP=pd.to_numeric(
            college.GRAD_DEBT_MDN_SUPP, errors='coerce'))
    .select_dtypes('number')
    .pipe(remove_binary_cols))


# In[82]:


college_n.max().head()


# In[83]:


college_n.eq(college_n.max()).head()


# In[84]:


has_row_max = (college_n.eq(college_n.max())
                        .any(axis='columns')
)
has_row_max.head()


# In[85]:


college_n.shape


# In[86]:


has_row_max.sum()


# ### 小編補充：

# In[87]:


s = pd.Series([0, 1, 0, 1, 0])
s.cumsum()


# In[88]:


college_n.eq(college_n.max()).cumsum()


# In[89]:


college_n.eq(college_n.max()).cumsum().cumsum()


# In[90]:


has_row_max2 = (college_n.eq(college_n.max()).cumsum() 
                                             .cumsum() 
                                             .eq(1) 
                                             .any(axis='columns'))


# In[91]:


has_row_max2.head()


# In[92]:


has_row_max2.sum()


# In[93]:


idxmax_cols = has_row_max2[has_row_max2].index
idxmax_cols


# In[94]:


set(college_n.idxmax().unique()) == set(idxmax_cols)


# In[95]:


def idx_max(df):
    has_row_max = (df.eq(df.max())
                     .cumsum()
                     .cumsum()
                     .eq(1)
                     .any(axis='columns'))
    return has_row_max[has_row_max].index

idx_max(college_n)


# In[96]:


def idx_max(df):
    has_row_max = (df.eq(df.max())
                     .cumsum()
                     .cumsum()
                     .eq(1)
                     .any(axis='columns')
                     [lambda df_: df_]
                     .index
    )
    return has_row_max


# In[97]:


get_ipython().run_line_magic('timeit', 'college_n.idxmax().values')


# In[98]:


get_ipython().run_line_magic('timeit', 'idx_max(college_n)')


# ## 8.8 尋找最常見的欄位最大值

# In[99]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()


# In[100]:


highest_percentage_race = college_ugds.idxmax(axis='columns')
highest_percentage_race.head()


# In[101]:


highest_percentage_race.value_counts(normalize=True)


# In[102]:


(college_ugds
    [highest_percentage_race == 'UGDS_BLACK']
    .drop(columns='UGDS_BLACK')
    .idxmax(axis='columns')
    .value_counts(normalize=True)
)

