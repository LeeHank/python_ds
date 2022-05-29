#!/usr/bin/env python
# coding: utf-8

# # 第11章：時間序列分析

# ## 11.1 了解Python和Pandas日期工具的區別

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10,'max_colwidth', 12)


# In[2]:


import datetime
date = datetime.date(year=2022, month=6, day=7)
time = datetime.time(hour=12, minute=30, second=19, microsecond=463198)
dt = datetime.datetime(year=2022, month=6, day=7, hour=12, minute=30, second=19,
                       microsecond=463198)
print(f'date is {date}')


# In[3]:


print(f'time is {time}')


# In[4]:


print(f'datetime is {dt}')


# In[5]:


td = datetime.timedelta(weeks=2, days=5, hours=10,
                        minutes=20, seconds=6.73,
                        milliseconds=99, microseconds=8)
td


# In[6]:


print(f'new date is {date+td}')


# In[7]:


print(f'new datetime is {dt+td}')


# In[8]:


# time + td


# In[9]:


pd.Timestamp(year=2021, month=12, day=21, hour=5,
             minute=10, second=8, microsecond=99)


# In[10]:


pd.Timestamp('2016/1/10')


# In[11]:


pd.Timestamp('2014-5/10')


# In[12]:


pd.Timestamp('Jan 3, 2019 20:45.56')


# In[13]:


pd.Timestamp('2016-01-05T05:34:43.123456789')


# In[14]:


pd.Timestamp(500)


# In[15]:


pd.Timestamp(5000, unit='D')


# In[16]:


pd.to_datetime('2015-5-13')


# In[17]:


pd.to_datetime('2015-13-5', dayfirst=True)


# In[18]:


pd.to_datetime('Start Date: Sep 30, 2017 Start Time: 1:30 pm',
               format='Start Date: %b %d, %Y Start Time: %I:%M %p')


# In[19]:


pd.to_datetime(100, unit='D', origin='2013-1-1')


# In[20]:


s = pd.Series([10, 100, 1000, 10000])
pd.to_datetime(s, unit='D')


# In[21]:


s = pd.Series(['12-5-2015', '14-1-2013', '20/12/2017', '40/23/2017'])
pd.to_datetime(s, dayfirst=True, errors='coerce')


# In[22]:


pd.to_datetime(['Aug 3 1999 3:45:56', '10/31/2017'])


# In[23]:


pd.Timedelta('12 days 5 hours 3 minutes 123456789 nanoseconds')


# In[24]:


pd.Timedelta(days=5, minutes=7.34)


# In[25]:


pd.Timedelta(100, unit='W')


# In[26]:


pd.to_timedelta('67:15:45.454')


# In[27]:


s = pd.Series([10, 100])
pd.to_timedelta(s, unit='s')


# In[28]:


time_strings = ['2 days 24 minutes 89.67 seconds', '00:45:23.6']
pd.to_timedelta(time_strings)


# In[29]:


pd.Timedelta('12 days 5 hours 3 minutes') * 2


# In[30]:


(pd.Timestamp('1/1/2022') + pd.Timedelta('12 days 5 hours 3 minutes') * 2)


# In[31]:


td1 = pd.to_timedelta([10, 100], unit='s')
td2 = pd.to_timedelta(['3 hours', '4 hours'])
td1 + td2


# In[32]:


pd.Timedelta('12 days') / pd.Timedelta('3 days')


# In[33]:


ts = pd.Timestamp('2021-10-1 4:23:23.9')
ts.ceil('h')


# In[34]:


ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second


# In[35]:


ts.dayofweek, ts.dayofyear, ts.daysinmonth


# In[36]:


ts.to_pydatetime()


# In[37]:


td = pd.Timedelta(125.8723, unit='h')
td


# In[38]:


td.round('min')


# In[39]:


td.components


# In[40]:


td.total_seconds()


# ## 11.2 對時間序列切片

# In[41]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


# In[42]:


mem_cat = crime.memory_usage().sum()
mem_obj = (crime
   .astype({'OFFENSE_TYPE_ID':'object',
            'OFFENSE_CATEGORY_ID':'object',
           'NEIGHBORHOOD_ID':'object'}) 
   .memory_usage(deep=True)
   .sum()
)
mb = 2 ** 20
round(mem_cat / mb, 1), round(mem_obj / mb, 1)


# In[43]:


crime = crime.set_index('REPORTED_DATE')
crime


# In[44]:


crime.index[:2]


# In[45]:


crime.loc['2016-05-12 16:45:00']


# In[46]:


crime.loc['2016-05-12']


# In[47]:


crime.loc['2016-05'].shape


# In[48]:


crime.loc['2016'].shape


# In[49]:


crime.loc['2016-05-12 03'].shape


# In[50]:


crime.loc['Dec 2015'].sort_index()


# In[51]:


crime.loc['2016 Sep, 15'].shape


# In[52]:


crime.loc['21st October 2014 05'].shape


# In[53]:


crime.loc['2015-3-4':'2016-1-1'].sort_index()


# In[54]:


crime.loc['2015-3-4 22':'2016-1-1 11:22:00'].sort_index()


# In[55]:


get_ipython().run_line_magic('timeit', "crime.loc['2015-3-4':'2016-1-1']")


# In[56]:


crime_sort = crime.sort_index()
get_ipython().run_line_magic('timeit', "crime_sort.loc['2015-3-4':'2016-1-1']")


# ## 11.3 過濾包含時間資料的欄位

# In[57]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


# In[58]:


(crime
    [crime.REPORTED_DATE == '2016-05-12 16:45:00']
)


# In[59]:


(crime
    [crime.REPORTED_DATE == '2016-05-12']
)


# In[60]:


(crime
    [crime.REPORTED_DATE.dt.date == '2016-05-12']
)


# In[61]:


(crime[crime.REPORTED_DATE.between(left='2016-05-12', right='2016-05-13')])


# In[62]:


(crime[crime.REPORTED_DATE.between('2016-05', '2016-06')].shape)


# In[63]:


(crime[crime.REPORTED_DATE.between('2016', '2017')].shape)


# In[64]:


(crime[crime.REPORTED_DATE.between('2016-05-12 03', '2016-05-12 04')].shape)


# In[65]:


(crime[crime.REPORTED_DATE.between('2016 Sep, 15', '2016 Sep, 16')].shape)


# In[66]:


(crime[crime.REPORTED_DATE.between('21st October 2014 05', 
                                   '21st October 2014 06')].shape)


# In[67]:


(crime[crime.REPORTED_DATE.between('2015-3-4 ','2016-1-1 23:59:59')].shape)


# In[68]:


(crime
    [crime.REPORTED_DATE.between(
         '2015-3-4 22','2016-1-1 11:22:00')]
    .shape
)


# In[69]:


lmask = crime.REPORTED_DATE >= '2015-3-4 22'
rmask = crime.REPORTED_DATE <= '2016-1-1 11:22:00'
crime[lmask & rmask].shape


# In[70]:


ctseries = crime.set_index('REPORTED_DATE')
get_ipython().run_line_magic('timeit', "ctseries.loc['2015-3-4':'2016-1-1']")


# In[71]:


get_ipython().run_line_magic('timeit', "crime[crime.REPORTED_DATE.between('2015-3-4','2016-1-1')]")


# ## 11.4 僅適用於DatetimeIndex的方法

# In[72]:


crime = (pd.read_hdf('data/crime.h5', 'crime').set_index('REPORTED_DATE'))
type(crime.index)


# In[73]:


crime.between_time('2:00', '5:00', include_end=False)


# In[74]:


import datetime
crime.between_time(datetime.time(2,0), datetime.time(5,0), include_end=False)


# In[75]:


crime.at_time('5:47')


# In[76]:


crime_sort = crime.sort_index()
crime_sort.first(offset = pd.offsets.MonthBegin(6))


# In[77]:


crime_sort.first(pd.offsets.MonthEnd(6))


# In[78]:


first_date = crime_sort.index[0]
first_date


# In[79]:


first_date + pd.offsets.MonthBegin(6)


# In[80]:


first_date + pd.offsets.MonthEnd(6)


# In[81]:


step4 = crime_sort.first(pd.offsets.MonthEnd(6))
end_dt = crime_sort.index[0] + pd.offsets.MonthEnd(6)
step4_internal = crime_sort[:end_dt]
step4.equals(step4_internal)


# In[82]:


crime_sort.first(pd.offsets.MonthBegin(6, normalize=True))


# In[83]:


crime_sort.loc[:'2012-06']


# In[84]:


crime_sort.first('5D') 


# In[85]:


crime_sort.first('5B') 


# In[86]:


crime_sort.first('7W') 


# In[87]:


crime_sort.first('3QS') 


# In[88]:


crime_sort.first('A') 


# In[89]:


dt = pd.Timestamp('2012-1-16 13:40')
dt + pd.DateOffset(months=1)


# In[90]:


do = pd.DateOffset(years=2, months=5, days=3, hours=8, seconds=10)
pd.Timestamp('2012-1-22 03:22') + do


# ## 11.5 依據時間區段重新分組

# In[91]:


crime_sort = (pd.read_hdf('data/crime.h5', 'crime') 
                .set_index('REPORTED_DATE') 
                .sort_index())


# In[92]:


crime_sort.resample('W')


# In[93]:


(crime_sort
    .resample('W')
    .size()
)


# In[94]:


len(crime_sort.loc[:'2012-1-8'])


# In[95]:


len(crime_sort.loc['2012-1-9':'2012-1-15'])


# In[96]:


(crime_sort
    .resample('W-THU')
    .size()
)


# In[97]:


weekly_crimes = (crime_sort.groupby(pd.Grouper(freq='W')) 
                           .size())
weekly_crimes


# In[98]:


r = crime_sort.resample('W')
[attr for attr in dir(r) if attr[0].islower()]


# In[99]:


crime = pd.read_hdf('data/crime.h5', 'crime')
weekly_crimes2 = crime.resample('W', on='REPORTED_DATE').size()
weekly_crimes2.equals(weekly_crimes)


# In[100]:


weekly_crimes_gby2 = (crime.groupby(pd.Grouper(key='REPORTED_DATE', freq='W'))
                           .size())
weekly_crimes2.equals(weekly_crimes)


# In[101]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 4))
weekly_crimes.plot(title='All Denver Crimes', ax=ax)


# ## 11.6 分組彙總同一時間單位的多個欄位

# In[102]:


crime = (pd.read_hdf('data/crime.h5', 'crime') 
           .set_index('REPORTED_DATE') 
           .sort_index())


# In[103]:


(crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[104]:


(crime
    .resample('QS')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[105]:


(crime
   .loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']]
   .sum()
)


# In[106]:


(crime
    .groupby(pd.Grouper(freq='Q')) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[107]:


fig, ax = plt.subplots(figsize=(16, 4))
(crime
    .groupby(pd.Grouper(freq='Q')) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .plot(color=['black', 'lightgrey'], ax=ax,
          title='Denver Crimes and Traffic Accidents')
)


# In[108]:


(crime
    .resample('Q')
    .sum()
)


# In[109]:


(crime_sort.resample('QS-MAR')
           ['IS_CRIME', 'IS_TRAFFIC'] 
           .sum())


# In[110]:


crime_begin = (crime.resample('Q')
                    ['IS_CRIME', 'IS_TRAFFIC']
                    .sum()
                    .iloc[0])


# In[111]:


fig, ax = plt.subplots(figsize=(16, 4))
(crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .div(crime_begin)
    .sub(1)
    .round(2)
    .mul(100)
    .plot.bar(color=['black', 'lightgrey'], ax=ax,
              title='Denver Crimes and Traffic Accidents % Increase')
)


# ## 11.7 案例演練：以『星期幾』來統計犯罪率

# In[112]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime


# In[113]:


(crime['REPORTED_DATE']
      .dt.day_name() 
      .value_counts())


# In[114]:


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday']
title = 'Denver Crimes and Traffic Accidents per Weekday'
fig, ax = plt.subplots(figsize=(6, 4))
(crime['REPORTED_DATE'].dt.day_name() 
                       .value_counts()
                       .reindex(days)
                       .plot.barh(title=title, ax=ax))         


# In[115]:


(crime
   ['REPORTED_DATE']
   .dt.day_name() 
   .value_counts()
   .loc[days]
)


# In[116]:


title = 'Denver Crimes and Traffic Accidents per Year'
fig, ax = plt.subplots(figsize=(6, 4))
(crime['REPORTED_DATE'].dt.year 
                       .value_counts()
                       .sort_index()
                       .plot.barh(title=title, ax=ax)
)
              


# In[117]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.day_name().rename('day')])
    .size()
)


# In[118]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.day_name().rename('day')])
    .size()
    .unstack('day')
)


# In[119]:


(crime
    .assign(year=crime.REPORTED_DATE.dt.year,
            day=crime.REPORTED_DATE.dt.day_name())
    .pipe(lambda df_: pd.crosstab(df_.year, df_.day))
)


# In[120]:


criteria = crime['REPORTED_DATE'].dt.year == 2017
crime.loc[criteria, 'REPORTED_DATE'].dt.dayofyear.max()


# In[121]:


crime_pct = (crime
   ['REPORTED_DATE']
   .dt.dayofyear.le(272) 
   .groupby(crime.REPORTED_DATE.dt.year) 
   .mean()
   .round(3)
)

crime_pct


# In[122]:


crime_pct.loc[2012:2016].median()


# In[123]:


def update_2017(df_):
    df_.loc[2017] = (df_.loc[2017]
                        .div(.748) 
                        .astype('int'))
    return df_

(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.day_name().rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
)


# In[124]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
table = (crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.day_name().rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
)
sns.heatmap(table, cmap='Greys', ax=ax)             


# In[125]:


denver_pop = pd.read_csv('data/denver_pop.csv', index_col='Year')
denver_pop


# In[126]:


den_100k = denver_pop.div(100_000).squeeze()
den_100k


# In[127]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.day_name().rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
) / den_100k


# In[128]:


den_100k = denver_pop.div(100_000).squeeze()
normalized = (crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.day_name().rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
    .div(den_100k, axis='index')
    .astype(int)
)
normalized


# In[129]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(normalized, cmap='Greys', ax=ax)               


# In[130]:


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday']
crime_type = 'auto-theft'
normalized = (crime
    .query('OFFENSE_CATEGORY_ID == @crime_type')
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.day_name().rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
    .div(den_100k, axis='index')
    .astype(int)
)
normalized


# ## 11.8 使用匿名函式來分組

# In[131]:


crime = (pd.read_hdf('data/crime.h5', 'crime') 
           .set_index('REPORTED_DATE') 
           .sort_index()
)


# In[132]:


common_attrs = (set(dir(crime.index)) & set(dir(pd.Timestamp)))
[attr for attr in common_attrs if attr[0] != '_']


# In[133]:


crime.index.day_name().value_counts()


# In[134]:


(crime
   .groupby(lambda idx: idx.day_name()) 
   ['IS_CRIME', 'IS_TRAFFIC']
   .sum()    
)


# In[135]:


funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime
    .groupby(funcs) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .unstack()
)


# In[136]:


funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime
    .groupby(funcs) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .unstack()
    .style.highlight_max(color='lightgrey')
)


# ## 11.9 使用Timestamp與另一欄位來分組

# In[137]:


employee = pd.read_csv('data/employee.csv',
    parse_dates=['JOB_DATE', 'HIRE_DATE'],
    index_col='HIRE_DATE')
employee


# In[138]:


(employee
    .groupby('GENDER')
    ['BASE_SALARY']
    .mean()
    .round(-2)
)


# In[139]:


(employee
    .resample('10AS')
    ['BASE_SALARY']
    .mean()
    .round(-2)    
)


# In[140]:


(employee
   .groupby('GENDER')
   .resample('10AS')
   ['BASE_SALARY'] 
   .mean()
   .round(-2)
)


# In[141]:


(employee
   .groupby('GENDER')
   .resample('10AS')
   ['BASE_SALARY'] 
   .mean()
   .round(-2)
   .unstack('GENDER')
)


# In[142]:


employee[employee['GENDER'] == 'Male'].index.min()


# In[143]:


employee[employee['GENDER'] == 'Female'].index.min()


# In[144]:


(employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
)


# In[145]:


(employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
   .unstack('GENDER')
)


# In[146]:


sal_final = (employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
   .unstack('GENDER')
)
years = sal_final.index.year
years_right = years + 9
sal_final.index = years.astype(str) + '-' + years_right.astype(str)
sal_final


# In[147]:


cuts = pd.cut(employee.index.year, bins=5, precision=0)
cuts.categories.values


# In[148]:


(employee
    .groupby([cuts, 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .unstack('GENDER')
    .round(-2)
)

