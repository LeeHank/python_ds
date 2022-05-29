#!/usr/bin/env python
# coding: utf-8

# # 第5章：探索式資料分析

# ## 摘要統計資訊

# In[1]:


import pandas as pd
import numpy as np
# pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)

fueleco = pd.read_csv('data/vehicles.csv.zip')
fueleco


# In[5]:


fueleco.describe().T


# In[6]:


fueleco.describe(include=object).T


# ## 5.2 轉換欄位的資料型別

# In[8]:


fueleco.dtypes


# In[9]:


fueleco.dtypes.value_counts()


# In[10]:


fueleco.select_dtypes('int64').describe().T


# In[11]:


np.iinfo(np.int8)


# In[12]:


np.iinfo(np.int16)


# In[13]:


fueleco[['city08', 'comb08']].info()


# In[14]:


(fueleco
  [['city08', 'comb08']]
  .assign(city08=fueleco.city08.astype(np.int16),
          comb08=fueleco.comb08.astype(np.int16))
  .info()
)


# In[15]:


fueleco.make.nunique()


# In[16]:


fueleco.model.nunique()


# In[17]:


fueleco[['make']].info(memory_usage='deep')


# In[18]:


(fueleco
  [['make']]
  .assign(make=fueleco.make.astype('category'))
  .info()
)


# In[19]:


fueleco[['model']].info(memory_usage='deep')


# In[20]:


(fueleco
  [['model']]
  .assign(model=fueleco.model.astype('category'))
  .info()
)


# ## 5.3 資料轉換與缺失值處理

# In[21]:


fueleco.select_dtypes(object).columns


# In[22]:


fueleco.drive.nunique()


# In[23]:


fueleco.drive.sample(5, random_state=42)


# In[24]:


fueleco.drive.isna().sum()


# In[25]:


fueleco.drive.isna().mean() * 100


# In[26]:


fueleco.drive.value_counts()


# In[27]:


fueleco.drive.value_counts(dropna=False)


# In[28]:


top_n = fueleco.make.value_counts().index[:6]
(fueleco
   .assign(make=fueleco.make.where(
              fueleco.make.isin(top_n), 'Other'))
   .make
   .value_counts()
)


# In[29]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
top_n = fueleco.make.value_counts().index[:6]
(fueleco     
   .assign(make=fueleco.make.where(
              fueleco.make.isin(top_n),
              'Other'))
   .make
   .value_counts()
   .plot.bar(ax=ax)
)


# In[30]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 8))
top_n = fueleco.make.value_counts().index[:6]
sns.countplot(y='make',     
    data= (fueleco
        .assign(make=fueleco.make.where(
                fueleco.make.isin(top_n),
                'Other'))
  )
)


# In[31]:


fueleco.rangeA.value_counts()


# In[32]:


(fueleco.rangeA.str.extract(r'([^0-9.])')
    .dropna()
    .apply(lambda row: ''.join(row), axis=1)
    .value_counts()
)


# In[33]:


set(fueleco.rangeA.apply(type))


# In[34]:


(fueleco
  .rangeA
  .fillna('0')
  .str.replace('-', '/')
  .str.split('/', expand=True)
  .astype(float)
  .mean(axis=1)
)


# In[35]:


(fueleco
  .rangeA
  .fillna('0')
  .str.replace('-', '/')
  .str.split('/', expand=True)
  .astype(float)
  .mean(axis=1)
  .pipe(lambda ser_: pd.cut(ser_, 10))
  .value_counts()
)


# In[36]:


# (fueleco
#   .rangeA
#   .fillna('0')
#   .str.replace('-', '/')
#   .str.split('/', expand=True)
#   .astype(float)
#   .mean(axis=1)
#   .pipe(lambda ser_: pd.qcut(ser_, 10))
#   .value_counts()
# )


# In[37]:


(fueleco
  .city08
  .pipe(lambda ser: pd.qcut(ser, q=10))
  .value_counts()
)


# ## 5.4 檢視連續資料的分佈狀況

# In[38]:


fueleco.select_dtypes('number')


# In[39]:


fueleco.city08.sample(5, random_state=42)


# In[40]:


fueleco.city08.isna().sum()


# In[41]:


fueleco.city08.isna().mean() * 100


# In[42]:


fueleco.city08.describe()


# In[43]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fueleco.city08.hist(ax=ax)


# In[44]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fueleco.city08.hist(ax=ax, bins=30)


# In[45]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.distplot(fueleco.city08, rug=True, ax=ax)


# In[46]:


fig, axs = plt.subplots(nrows=3, figsize=(10, 8))
sns.boxplot(fueleco.city08, ax=axs[0])
sns.violinplot(fueleco.city08, ax=axs[1])
sns.boxenplot(fueleco.city08, ax=axs[2])    


# In[47]:


from scipy import stats
stats.kstest(fueleco.city08, cdf='norm')


# In[48]:


from scipy import stats
fig, ax = plt.subplots(figsize=(10, 8))
stats.probplot(fueleco.city08, plot=ax) 


# ## 5.5 檢視不同分類的資料分佈

# In[49]:


fueleco.make


# In[50]:


mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
fueleco[mask].groupby('make').city08.agg(['mean', 'std'])


# In[51]:


g = sns.catplot(x='make', y='city08', 
                data=fueleco[mask], kind='box')    


# In[52]:


mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
(fueleco[mask].groupby('make').city08.count())


# In[53]:


g = sns.catplot(x='make', y='city08', 
                data=fueleco[mask], kind='box')
sns.swarmplot(x='make', y='city08',    
              data=fueleco[mask], color='k', size=1, ax=g.ax)


# In[54]:


g = sns.catplot(x='make', y='city08', 
                data=fueleco[mask], kind='box',
                col='year', col_order=[2012, 2014, 2016, 2018],
                col_wrap=2)


# In[55]:


g = sns.catplot(x='make', y='city08', 
                data=fueleco[mask], kind='box',
                hue='year', hue_order=[2012, 2014, 2016, 2018])


# In[56]:


mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
(fueleco
  [mask]
  .groupby('make')
  .city08
  .agg(['mean', 'std'])
  .style.background_gradient(cmap='RdBu', axis=0)
)


# ## 5.6 比較連續欄位間的關聯性

# In[57]:


fueleco.city08.cov(fueleco.highway08)


# In[58]:


fueleco.city08.cov(fueleco.comb08)


# In[59]:


fueleco.city08.cov(fueleco.cylinders)


# In[60]:


fueleco.city08.corr(fueleco.highway08)


# In[61]:


fueleco.city08.corr(fueleco.cylinders)


# In[62]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(8,8))
corr = fueleco[['city08', 'highway08', 'cylinders']].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask,
            fmt='.2f', annot=True, ax=ax, cmap='RdBu', vmin=-1, vmax=1,
            square=True)


# In[63]:


fig, ax = plt.subplots(figsize=(8,8))
fueleco.plot.scatter(x='city08', y='highway08', alpha=.1, ax=ax)


# In[64]:


fig, ax = plt.subplots(figsize=(8,8))
fueleco.plot.scatter(x='city08', y='cylinders', alpha=.1, ax=ax)


# In[65]:


fueleco.cylinders.isna().sum()


# In[66]:


fig, ax = plt.subplots(figsize=(8,8))
(fueleco
 .assign(cylinders=fueleco.cylinders.fillna(0))
 .plot.scatter(x='city08', y='cylinders', alpha=.1, ax=ax))


# In[67]:


res = sns.lmplot(x='city08', y='highway08', data=fueleco)


# In[68]:


fueleco.city08.corr(fueleco.highway08*2)


# In[69]:


fueleco.city08.cov(fueleco.highway08*2)


# In[70]:


res = sns.relplot(x='city08', y='highway08',
                  data=fueleco.assign(
                      cylinders=fueleco.cylinders.fillna(0)),
                  hue='year', size='barrels08', alpha=.5, height=8)


# In[71]:


res = sns.relplot(x='city08', y='highway08',
  data=fueleco.assign(
  cylinders=fueleco.cylinders.fillna(0)),
  hue='year', size='barrels08', alpha=.5, height=8,
  col='make', col_order=['Ford', 'Tesla'])


# In[72]:


fueleco.city08.corr(fueleco.barrels08, method='spearman')


# ## 5.7 比較分類欄位的關聯性

# In[73]:


def generalize(ser, match_name, default):
    seen = None
    for match, name in match_name:
        mask = ser.str.contains(match)
        if seen is None:
            seen = mask
        else:
            seen |= mask
        ser = ser.where(~mask, name)
    ser = ser.where(seen, default)
    return ser


# In[74]:


makes = ['Ford', 'Tesla', 'BMW', 'Toyota']
data = (fueleco
   [fueleco.make.isin(makes)]
   .assign(SClass=lambda df_: generalize(df_.VClass,
    [('Seaters', 'Car'), ('Car', 'Car'), ('Utility', 'SUV'),
     ('Truck', 'Truck'), ('Van', 'Van'), ('van', 'Van'),
     ('Wagon', 'Wagon')], 
    'other'))
)


# In[75]:


data.groupby(['make', 'SClass']).size().unstack()


# In[76]:


pd.crosstab(data.make, data.SClass)


# In[77]:


pd.crosstab([data.year, data.make], [data.SClass, data.VClass])


# In[78]:


import scipy.stats as ss
import numpy as np
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[79]:


cramers_v(data.make, data.SClass)


# In[80]:


data.make.corr(data.SClass, cramers_v)


# In[81]:


fig, ax = plt.subplots(figsize=(10,8))
(data.pipe(lambda df_: pd.crosstab(df_.make, df_.SClass))
     .plot.bar(ax=ax)
)


# In[82]:


res = sns.catplot(kind='count', x='make', hue='SClass', data=data)


# In[83]:


fig, ax = plt.subplots(figsize=(10,8))
(data
 .pipe(lambda df_: pd.crosstab(df_.make, df_.SClass))
 .pipe(lambda df_: df_.div(df_.sum(axis=1), axis=0))
 .plot.bar(stacked=True, ax=ax)
)


# ## 5.8 使用Profiling函式庫建立摘要報告

# In[84]:


pip install pandas-profiling


# In[ ]:


import pandas_profiling as pp
pp.ProfileReport(fueleco)


# In[ ]:


report = pp.ProfileReport(fueleco)
report.to_file('fuel.html')

