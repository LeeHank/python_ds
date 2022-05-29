#!/usr/bin/env python
# coding: utf-8

# # 第9章：透過分組來進行聚合、過濾和轉換

# In[65]:


df = pd.DataFrame({
    "col1": ["A", "A", "A", "A", "B", "B", "B", "B"],
    "col2": ["G", "Y", "G", "Y", "G", "Y", "G", "Y"],
    "col3": [1, 2, 3, 4, 5, 6, 7, 8],
    "col4": [4, 3, 2, 1, 3, 4, 9, 2]
})
df


# In[66]:


(
    df
        .groupby(["col1", "col2"], as_index = False)
        .agg({"col3": np.mean})
)


# In[67]:


(
    df
        .groupby(["col1", "col2"], as_index = False)
        .apply(lambda x: x.assign(new = (x.col3-x.col3.mean())/x.col3.std()))
)


# In[68]:


df.groupby(["col1", "col2"]).apply(lambda x: x.assign(n = len(x)))


# In[74]:


df.groupby(["col1", "col2"]).apply(lambda x: x.sort_values("col4", ascending = False).loc[:, ["col3", "col4"]]).reset_index()


# ## 9.1 進行簡單的分組及聚合運算

# In[31]:


import pandas as pd
import numpy as np
#pd.set_option('max_columns', 4, 'max_rows', 10)

flights = pd.read_csv('data/flights.csv')
flights.head()


# In[13]:


np.mean(np.array([1, 2, np.nan]), )


# In[16]:


get_ipython().run_line_magic('pinfo', 'np.percentile')


# In[26]:


flights.ARR_DELAY.quantile(0.5)


# In[25]:


flights.ARR_DELAY


# In[29]:


get_ipython().run_line_magic('pinfo', 'pd.Series.mean')


# In[39]:


(
    flights
        .groupby("AIRLINE")
        .loc[:, "ARR_DELAY"]
        .mean()
)


# In[35]:


(
    flights
        .groupby("AIRLINE")
        .agg(sd_arr_delay = pd.NamedAgg("ARR_DELAY", np.std))
        .reset_index()
)


# In[43]:


(
    flights
        .groupby("AIRLINE", as_index=False)
        .agg(sd_arr_delay = pd.NamedAgg("ARR_DELAY", "std"),
             mean_arr_delay = pd.NamedAgg("ARR_DELAY", lambda x: x.mean(skipna=True)),
            pr50_arr_delay = pd.NamedAgg("ARR_DELAY", lambda x: np.quantile(x, q = 0.5)))
        .assign(plus_one = lambda x: x.mean_arr_delay + 1)
        .assign(z_score = lambda x: (x.plus_one - x.plus_one.mean())/x.plus_one.std())
)


# In[30]:


(
    flights
        .groupby("AIRLINE", as_index=False)
        .agg(sd_arr_delay = pd.NamedAgg("ARR_DELAY", "std"),
             mean_arr_delay = pd.NamedAgg("ARR_DELAY", lambda x: x.mean(skipna=True)),
            pr50_arr_delay = pd.NamedAgg("ARR_DELAY", lambda x: x.quantile(0.5)))
        .assign(plus_one = lambda x: x.mean_arr_delay + 1)
        .assign(z_score = lambda x: (x.plus_one - x.plus_one.mean())/x.plus_one.std())
)


# In[2]:


(flights
     .groupby('AIRLINE')
     .agg({'ARR_DELAY':'mean'})
)


# In[3]:


(flights
     .groupby('AIRLINE')
     ['ARR_DELAY']
     .agg('mean')
)


# In[4]:


(flights
     .groupby('AIRLINE')
     ['ARR_DELAY']
     .agg(np.mean)
)


# In[5]:


(flights
     .groupby('AIRLINE')
     ['ARR_DELAY']
     .mean()
)


# In[6]:


# (flights
#    .groupby('AIRLINE')
#    ['ARR_DELAY']
#    .agg(np.sqrt)
# )


# ## 9.2 對多個欄位執行分組及聚合運算

# In[7]:


(flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    ['CANCELLED'] 
    .agg('sum')
)


# In[8]:


(flights
    .groupby(['AIRLINE', 'WEEKDAY']) 
    ['CANCELLED', 'DIVERTED']
    .agg(['sum', 'mean'])
)


# In[9]:


(flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)


# In[10]:


(flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg(sum_cancelled=pd.NamedAgg(column='CANCELLED', aggfunc='sum'),
         mean_cancelled=pd.NamedAgg(column='CANCELLED', aggfunc='mean'),
         size_cancelled=pd.NamedAgg(column='CANCELLED', aggfunc='size'),
         mean_air_time=pd.NamedAgg(column='AIR_TIME', aggfunc='mean'),
         var_air_time=pd.NamedAgg(column='AIR_TIME', aggfunc='var'))
)


# In[11]:


res = (flights.groupby(['ORG_AIR', 'DEST_AIR'])
              .agg({'CANCELLED':['sum', 'mean', 'size'],
                    'AIR_TIME':['mean', 'var']})
)
res.columns


# In[12]:


res_flat_column = res.columns.to_flat_index()
res_flat_column


# In[13]:


res.columns = ['_'.join(x) for x in res_flat_column]
res


# In[14]:


def flatten_cols(df):
    df.columns = ['_'.join(x) for x in df.columns.to_flat_index()]
    return df

res = (flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
    .pipe(flatten_cols)
)

res


# In[15]:


res = (flights
    .assign(ORG_AIR=flights.ORG_AIR.astype('category'))
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)
res


# In[16]:


res = (flights
    .assign(ORG_AIR=flights.ORG_AIR.astype('category'))
    .groupby(['ORG_AIR', 'DEST_AIR'], observed=True)
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)
res


# ## 9.3 分組後刪除MultiIndex

# In[17]:


flights = pd.read_csv('data/flights.csv')
airline_info = (flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    .agg({'DIST':['sum', 'mean'],
          'ARR_DELAY':['min', 'max']}) 
    .astype(int)
)
airline_info


# In[18]:


airline_info.columns.get_level_values(0)


# In[19]:


airline_info.columns.get_level_values(1)


# In[20]:


airline_info.columns.to_flat_index()


# In[21]:


airline_info.columns = ['_'.join(x) for x in
    airline_info.columns.to_flat_index()]

airline_info


# In[22]:


airline_info.reset_index()


# In[23]:


(flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    .agg(dist_sum=pd.NamedAgg(column='DIST', aggfunc='sum'),
         dist_mean=pd.NamedAgg(column='DIST', aggfunc='mean'),
         arr_delay_min=pd.NamedAgg(column='ARR_DELAY', aggfunc='min'),
         arr_delay_max=pd.NamedAgg(column='ARR_DELAY', aggfunc='max'))
    .astype(int)
    .reset_index()
)


# In[24]:


(flights
    .groupby(['AIRLINE'], as_index=False)
    ['DIST']
    .agg('mean')
    .round(0)
)


# ## 9.4 使用自訂的聚合函式來分組

# In[25]:


college = pd.read_csv('data/college.csv')
(college
    .groupby('STABBR')
    ['UGDS']
    .agg(['mean', 'std'])
    .round(0)
)


# In[26]:


def max_deviation(s):
    std_score = (s - s.mean()) / s.std()
    return std_score.abs().max()


# In[27]:


(college
    .groupby('STABBR')
    ['UGDS']
    .agg(max_deviation)
    .round(1)
)


# In[28]:


(college
    .groupby('STABBR')
    ['UGDS', 'SATVRMID', 'SATMTMID']
    .agg(max_deviation)
    .round(1)
)


# In[29]:


(college
    .groupby(['STABBR']) 
    ['UGDS'] 
    .agg([max_deviation, 'mean', 'std'])
    .round(1)
)


# In[30]:


max_deviation.__name__


# In[31]:


max_deviation.__name__ = 'Max Deviation'
(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATVRMID', 'SATMTMID'] 
    .agg([max_deviation, 'mean', 'std'])
    .round(1)
)


# ## 9.5 可接收多個參數的自訂聚合函式

# In[32]:


def pct_between_1_3k(s):
    return (s.between(1_000, 3_000)
             .mean()* 100)


# In[33]:


(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between_1_3k)
    .round(1)
)


# In[34]:


def pct_between(s, low, high):
    return s.between(low, high).mean() * 100


# In[35]:


(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between, 1_000, 10_000)
    .round(1)
)


# In[36]:


(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between, low=1_000, high=10_000)
    .round(1)
)


# In[37]:


def between_n_m(n, m):
    def wrapper(ser):
        return pct_between(ser, n, m)
    wrapper.__name__ = f'between_{n}_{m}'
    return wrapper

(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg([between_n_m(1_000, 10_000), 'max', 'mean'])
    .round(1)
)


# ## 9.6 深入了解groupby物件

# In[45]:


college = pd.read_csv('data/college.csv')
grouped = college.groupby(['STABBR', 'RELAFFIL'])
type(grouped)


# In[50]:


college.columns


# In[39]:


print([attr for attr in dir(grouped) if not
    attr.startswith('_')])


# In[51]:


grouped.ngroups


# In[52]:


groups = list(grouped.groups)
groups[:6]


# In[53]:


grouped.get_group(('FL', 1))


# In[54]:


from IPython.display import display
for name, group in grouped:
    print(name)
    display(group.head(3))


# In[44]:


for name, group in grouped:
    print(name)
    display(group)
    break


# In[45]:


grouped.head(1)


# In[46]:


grouped.nth([1, -1])


# ## 9.7 過濾特定的組別

# In[47]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
grouped = college.groupby('STABBR')
grouped.ngroups


# In[48]:


college['STABBR'].nunique()


# In[49]:


def check_minority(df, threshold):
    minority_pct = 1 - df['UGDS_WHITE']
    total_minority = (df['UGDS'] * minority_pct).sum()
    total_ugds = df['UGDS'].sum()
    total_minority_pct = total_minority / total_ugds
    return total_minority_pct > threshold


# In[50]:


college_filtered = grouped.filter(check_minority, threshold=.5)
college_filtered


# In[51]:


college.shape


# In[52]:


college_filtered.shape


# In[53]:


college_filtered['STABBR'].nunique()


# In[54]:


college_filtered_20 = grouped.filter(check_minority, threshold=.2)
college_filtered_20.shape


# In[55]:


college_filtered_20['STABBR'].nunique()


# In[56]:


college_filtered_70 = grouped.filter(check_minority, threshold=.7)
college_filtered_70.shape


# In[57]:


college_filtered_70['STABBR'].nunique()


# ## 9.8 分組轉換特定欄位的資料

# In[58]:


weight_loss = pd.read_csv('data/weight_loss.csv')
weight_loss.query('Month == "Jan"')


# In[59]:


def percent_loss(s):
    return ((s - s.iloc[0]) / s.iloc[0]) * 100


# In[60]:


(weight_loss
    .query('Name=="Bob" and Month=="Jan"')
    ['Weight']
    .pipe(percent_loss)
)


# In[61]:


(weight_loss
    .groupby(['Name', 'Month'])
    ['Weight'] 
    .transform(percent_loss)
)


# In[62]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Name=="Bob" and Month in ["Jan", "Feb"]')
)


# In[63]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
)


# In[64]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
)


# In[65]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
)


# In[66]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
    .style.highlight_min(axis=1,color='lightgrey')
)


# In[67]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
    .winner
    .value_counts()
)


# In[68]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)),
            Month=pd.Categorical(weight_loss.Month,
                  categories=['Jan', 'Feb', 'Mar', 'Apr'],
                  ordered=True))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
)


# ## 9.9 使用apply()計算加權平均數

# In[69]:


college = pd.read_csv('data/college.csv')
subset = ['UGDS', 'SATMTMID', 'SATVRMID']
college2 = college.dropna(subset=subset)
college.shape


# In[70]:


college2.shape


# In[71]:


def weighted_math_average(df):
    weighted_math = df['UGDS'] * df['SATMTMID']
    return int(weighted_math.sum() / df['UGDS'].sum())


# In[72]:


college2.groupby('STABBR').apply(weighted_math_average) 


# In[73]:


# (college2
#     .groupby('STABBR')
#     .agg(weighted_math_average)
# )


# In[74]:


# (college2
#     .groupby('STABBR')
#     ['SATMTMID'] 
#     .agg(weighted_math_average)
# )


# In[75]:


def weighted_average(df):
    weight_m = df['UGDS'] * df['SATMTMID']
    weight_v = df['UGDS'] * df['SATVRMID']
    wm_avg = weight_m.sum() / df['UGDS'].sum()
    wv_avg = weight_v.sum() / df['UGDS'].sum()
    data = {'w_math_avg': wm_avg,
            'w_verbal_avg': wv_avg,
            'math_avg': df['SATMTMID'].mean(),
            'verbal_avg': df['SATVRMID'].mean(),
            'count': len(df)
    }
    return pd.Series(data)


# In[76]:


weighted_average(college2)


# In[77]:


(college2
    .groupby('STABBR')
    .apply(weighted_average)
    .astype(int)
)


# In[78]:


from scipy.stats import gmean, hmean
def calculate_means(df):
    df_means = pd.DataFrame(index=['Arithmetic', 'Weighted',
                                   'Geometric', 'Harmonic'])
    cols = ['SATMTMID', 'SATVRMID']
    for col in cols:
        arithmetic = df[col].mean()
        weighted = np.average(df[col], weights=df['UGDS'])
        geometric = gmean(df[col])
        harmonic = hmean(df[col])
        df_means[col] = [arithmetic, weighted,
                         geometric, harmonic]
    df_means['count'] = len(df)
    return df_means.astype(int)
(college2
    .groupby('STABBR')
    .apply(calculate_means)
)


# ## 9.10 以連續變化的數值進行分組

# In[79]:


flights = pd.read_csv('data/flights.csv')
flights


# In[80]:


bins = [-np.inf, 200, 500, 1000, 2000, np.inf]
cuts = pd.cut(flights['DIST'], bins=bins)
cuts


# In[81]:


cuts.value_counts()


# In[82]:


(flights
    .groupby(cuts)
    ['AIRLINE']
    .value_counts(normalize=True) 
    .round(3)
)


# In[83]:


(flights
  .groupby(cuts)
  ['AIR_TIME']
  .quantile(q=[.25, .5, .75]) 
  .div(60)
  .round(2)
)


# ## 9.11 案例演練：計算城市之間的航班總數

# In[84]:


flights = pd.read_csv('data/flights.csv')
flights_ct = flights.groupby(['ORG_AIR', 'DEST_AIR']).size()
flights_ct


# In[85]:


flights_ct.loc[[('ATL', 'IAH'), ('IAH', 'ATL')]]


# In[86]:


f_part3 = (flights  
  [['ORG_AIR', 'DEST_AIR']] 
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
)
f_part3


# In[87]:


rename_dict = {0:'AIR1', 1:'AIR2'}  
(flights    
  [['ORG_AIR', 'DEST_AIR']]
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
  .rename(columns=rename_dict)
  .groupby(['AIR1', 'AIR2'])
  .size()
)


# In[88]:


(flights     
  [['ORG_AIR', 'DEST_AIR']]
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
  .rename(columns=rename_dict)
  .groupby(['AIR1', 'AIR2'])
  .size()
  .loc[('ATL', 'IAH')]
)


# In[89]:


# (flights     
#   [['ORG_AIR', 'DEST_AIR']]
#   .apply(lambda ser:
#          ser.sort_values().reset_index(drop=True),
#          axis='columns')
#   .rename(columns=rename_dict)
#   .groupby(['AIR1', 'AIR2'])
#   .size()
#   .loc[('IAH', 'ATL')]
# )


# In[90]:


data_sorted = np.sort(flights[['ORG_AIR', 'DEST_AIR']])
data_sorted[:10]


# In[91]:


flights_sort2 = pd.DataFrame(data_sorted, columns=['AIR1', 'AIR2'])
flights_sort2.equals(f_part3.rename(columns={0:'AIR1', 1:'AIR2'}))


# In[92]:


get_ipython().run_cell_magic('timeit', '', "flights_sort = (flights   # doctest: +SKIP\n    [['ORG_AIR', 'DEST_AIR']] \n   .apply(lambda ser:\n         ser.sort_values().reset_index(drop=True),\n         axis='columns')\n)")


# In[93]:


get_ipython().run_cell_magic('timeit', '', "data_sorted = np.sort(flights[['ORG_AIR', 'DEST_AIR']])\nflights_sort2 = pd.DataFrame(data_sorted, columns=['AIR1', 'AIR2'])")


# ## 9.12 案例演練：尋找航班的連續準時記錄

# In[94]:


s = pd.Series([0, 1, 1, 0, 1, 1, 1, 0])
s


# In[95]:


s1 = s.cumsum()
s1


# In[96]:


s.mul(s1)


# In[97]:


s.mul(s1).diff()


# In[98]:


(s.mul(s.cumsum())
  .diff()
  .where(lambda x: x < 0))


# In[99]:


(s.mul(s.cumsum())
  .diff()
  .where(lambda x: x < 0)
  .ffill())


# In[100]:


(s.mul(s.cumsum())
  .diff()
  .where(lambda x: x < 0)
  .ffill()
  .add(s.cumsum(), fill_value=0))


# In[101]:


flights = pd.read_csv('data/flights.csv')
(flights.assign(ON_TIME=flights['ARR_DELAY'].lt(15).astype(int))
        [['AIRLINE', 'ORG_AIR', 'ON_TIME']])


# In[102]:


def max_streak(s):
    s1 = s.cumsum()
    return (s.mul(s1)
             .diff()
             .where(lambda x: x < 0) 
             .ffill()
             .add(s1, fill_value=0)
             .max())


# In[103]:


(flights
    .assign(ON_TIME=flights['ARR_DELAY'].lt(15).astype(int))
    .sort_values(['MONTH', 'DAY', 'SCHED_DEP']) 
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['ON_TIME'] 
    .agg(['mean', 'size', max_streak])
    .round(2)
)

