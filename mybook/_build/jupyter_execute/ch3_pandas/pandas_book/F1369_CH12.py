#!/usr/bin/env python
# coding: utf-8

# # 第12章：利用Matplotlib、Pandas和Seaborn進行資料視覺化

# ## 12.1 Matplotlib入門

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## 12.2 Matplotlib的物件導向指南

# In[2]:


import matplotlib.pyplot as plt
x = [-3, 5, 7]
y = [10, 2, 5]
fig = plt.figure(figsize=(15,3))
plt.plot(x, y)
plt.xlim(0, 10)
plt.ylim(-3, 8)
plt.xlabel('X Axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.suptitle('Figure Title', size=20, y=1.03)


# In[3]:


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython.core.display import display
fig = Figure(figsize=(15, 3))
FigureCanvas(fig)    
ax = fig.add_subplot(1,1,1)
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-3, 8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Line Plot')
fig.suptitle('Figure Title', size=20, y=1.03)
display(fig)


# In[4]:


fig, ax = plt.subplots(figsize=(15,3))
ax.plot(x, y)
ax.set(xlim=(0, 10), ylim=(-3, 8),
       xlabel='X axis', ylabel='Y axis',
       title='Line Plot')
fig.suptitle('Figure Title', size=20, y=1.03)


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


fig, ax = plt.subplots(nrows=1, ncols=1)       


# In[7]:


plot_objects = plt.subplots(nrows=1, ncols=1)
type(plot_objects)


# In[8]:


fig = plot_objects[0]
ax = plot_objects[1]


# In[9]:


figs, axs = plt.subplots(2, 4)


# In[10]:


axs


# In[11]:


type(fig)


# In[12]:


type(ax)


# In[13]:


fig.get_size_inches()


# In[14]:


fig.set_size_inches(14, 4)         
fig


# In[15]:


fig.axes


# In[16]:


ax.xaxis == ax.get_xaxis()


# In[17]:


ax.yaxis == ax.get_yaxis()


# In[18]:


fig.axes[0] is ax


# In[19]:


fig.set_facecolor('.7')
ax.set_facecolor('.5')
fig


# In[20]:


ax_children = ax.get_children()
ax_children


# In[21]:


spines = ax.spines
spines


# In[22]:


spine_left = spines['left']
spine_left.set_position(('outward', -100))
spine_left.set_linewidth(5)
spine_bottom = spines['bottom']
spine_bottom.set_visible(False)
fig


# In[23]:


ax.xaxis.grid(True, which='major', linewidth=2, color='black', linestyle='--')
ax.xaxis.set_ticks([.2, .4, .55, .93])
ax.xaxis.set_label_text('X Axis', family='Verdana', fontsize=15)
ax.set_ylabel('Y Axis', family='Gotham', fontsize=20)
ax.set_yticks([.1, .9])
ax.set_yticklabels(['point 1', 'point 9'], rotation=45)
fig


# In[24]:


ax.xaxis.properties()


# ## 12.3 用Matplotlib視覺化資料

# In[25]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 6, 'max_rows', 10, 'max_colwidth', 12)

alta = pd.read_csv('data/alta-noaa-1980-2019.csv')
alta


# In[26]:


data = (alta.assign(DATE=pd.to_datetime(alta.DATE))
            .set_index('DATE')
            .loc['2018-09':'2019-08']
            .SNWD)
data


# In[27]:


blue = '#99ddee'
white = '#ffffff'
fig, ax = plt.subplots(figsize=(12,4), linewidth=5, facecolor=blue)
ax.set_facecolor(blue)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', colors=white)
ax.tick_params(axis='y', colors=white)
ax.set_ylabel('Snow Depth (in)', color=white)
ax.set_title('2018-2019', color=white, fontweight='bold')
ax.fill_between(data.index, data, color=white)


# In[28]:


import matplotlib.dates as mdt
blue = '#99ddee'
white = '#ffffff'


# In[29]:


def plot_year(ax, data, years):
    ax.set_facecolor(blue)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', colors=white)
    ax.tick_params(axis='y', colors=white)
    ax.set_ylabel('Snow Depth (in)', color=white)
    ax.set_title(years, color=white, fontweight='bold')
    ax.fill_between(data.index, data, color=white)


# In[30]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
                        figsize=(16, 10), linewidth=5, facecolor=blue)

axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta
       .assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
                xy=(mdt.date2num(max_data.idxmax()), max_val), 
                color=white)

fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout()


# In[31]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
                        figsize=(16, 10), linewidth=5, facecolor=blue)
axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta.assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
       .interpolate()
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
   xy=(mdt.date2num(max_data.idxmax()), max_val), 
   color=white)
plt.tight_layout()

fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[32]:


(alta
    .assign(DATE=pd.to_datetime(alta.DATE))
    .set_index('DATE')
    .SNWD
    .to_frame()
    .assign(next=lambda df_:df_.SNWD.shift(-1),
            snwd_diff=lambda df_:df_.next-df_.SNWD)
    .pipe(lambda df_: df_[df_.snwd_diff.abs() > 50])
)


# In[33]:


def fix_gaps(ser, threshold=50):
    'Replace values where the shift is > threshold with nan'
    mask = (ser
       .to_frame()
       .assign(next=lambda df_:df_.SNWD.shift(-1),
               snwd_diff=lambda df_:df_.next-df_.SNWD)
       .pipe(lambda df_: df_.snwd_diff.abs() > threshold)
    )
    return ser.where(~mask, np.nan)


# In[34]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
                        figsize=(16, 10), linewidth=5, facecolor=blue)
axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta.assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
       .pipe(fix_gaps)
       .interpolate()
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
                xy=(mdt.date2num(max_data.idxmax()), max_val), 
                color=white)

fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# ### 小編補充

# In[35]:


s = pd.Series(range(5))
s


# In[36]:


s.where(s>0)


# In[37]:


s = pd.Series(range(5))
t = s>0
t


# In[38]:


s.where(t)


# ## 12.4 使用Pandas繪製基本圖形

# In[39]:


df = pd.DataFrame(index=['Atiya', 'Abbas', 'Cornelia', 'Stephanie', 'Monte'],
                  data={'Apples':[20, 10, 40, 20, 50],
                        'Oranges':[35, 40, 25, 19, 33]})


# In[40]:


df


# In[41]:


ax = df.plot.bar(figsize=(16,4))


# In[42]:


ax = df.plot.kde(figsize=(16,4))


# In[43]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
fig.suptitle('Two Variable Plots', size=20, y=1.02)
df.plot.line(ax=ax1, title='Line plot')
df.plot.scatter(x='Apples', y='Oranges', 
    ax=ax2, title='Scatterplot')
df.plot.bar(ax=ax3, title='Bar plot')


# In[44]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
fig.suptitle('One Variable Plots', size=20, y=1.02)
df.plot.kde(ax=ax1, title='KDE plot')
df.plot.box(ax=ax2, title='Boxplot')
df.plot.hist(ax=ax3, title='Histogram')


# In[45]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
df.sort_values('Apples').plot.line(x='Apples', y='Oranges', ax=ax1)
df.plot.bar(x='Apples', y='Oranges', ax=ax2)
df.plot.kde(x='Apples', ax=ax3)


# ## 12.5 視覺化航班資料集

# In[46]:


flights = pd.read_csv('data/flights.csv')


# In[47]:


cols = ['DIVERTED', 'CANCELLED', 'DELAYED']
(flights
    .assign(DELAYED=flights['ARR_DELAY'].ge(15).astype(int),
            ON_TIME=lambda df_:1 - df_[cols].any(axis=1))
    .select_dtypes(int)
    .sum()
)


# In[48]:


fig, ax_array = plt.subplots(2, 3, figsize=(18,8))
(ax1, ax2, ax3), (ax4, ax5, ax6) = ax_array
fig.suptitle('2015 US Flights - Univariate Summary', size=20)
ac = flights['AIRLINE'].value_counts()
ac.plot.barh(ax=ax1, title='Airline')
(flights
    ['ORG_AIR']
    .value_counts()
    .plot.bar(ax=ax2, rot=0, title='Origin City')
)
(flights
    ['DEST_AIR']
    .value_counts()
    .head(10)
    .plot.bar(ax=ax3, rot=0, title='Destination City')
)
(flights
    .assign(DELAYED=flights['ARR_DELAY'].ge(15).astype(int),
            ON_TIME=lambda df_:1 - df_[cols].any(axis=1))
    [['DIVERTED', 'CANCELLED', 'DELAYED', 'ON_TIME']]
    .sum()
    .plot.bar(ax=ax4, rot=0,
         log=True, title='Flight Status')
)
flights['DIST'].plot.kde(ax=ax5, xlim=(0, 3000), title='Distance KDE')
flights['ARR_DELAY'].plot.hist(ax=ax6, title='Arrival Delay', range=(0,200)
)


# In[49]:


df_date = (flights
    [['MONTH', 'DAY']]
    .assign(YEAR=2015,
            HOUR=flights['SCHED_DEP'] // 100,
            MINUTE=flights['SCHED_DEP'] % 100)
)
df_date


# In[50]:


flight_dep = pd.to_datetime(df_date)
flight_dep


# In[51]:


flights.index = flight_dep
fc = flights.resample('W').size()
fc.plot.line(figsize=(12,6), title='Flights per Week', grid=True)


# In[52]:


def interp_lt_n(df_, n=600):
    return (df_
        .where(df_ > n)
        .interpolate(limit_direction='both')
)
fig, ax = plt.subplots(figsize=(16,4))
data = (flights
    .resample('W')
    .size()
)
(data
    .pipe(interp_lt_n)
    .iloc[1:-1]
    .plot.line(ax=ax)
)
mask = data<600
(data
     .pipe(interp_lt_n)[mask]
     .plot.line(color='.8', linewidth=10)
) 
ax.annotate(xy=(.8, .55), xytext=(.8, .77),
            xycoords='axes fraction', s='missing data',
            ha='center', size=20, arrowprops=dict())
ax.set_title('Flights per Week (Interpolated Missing Data)')


# In[53]:


fig, ax = plt.subplots(figsize=(16,4))
(flights
    .groupby('DEST_AIR')
    ['DIST'] 
    .agg(['mean', 'count']) 
    .query('count > 100') 
    .sort_values('mean') 
    .tail(10) 
    .plot.bar(y='mean', rot=0, legend=False, ax=ax,
              title='Average Distance per Destination')
)


# In[54]:


fig, ax = plt.subplots(figsize=(8,6))
(flights
    .reset_index(drop=True)
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .plot.scatter(x='DIST', y='AIR_TIME', ax=ax, alpha=.1, s=1)
)


# In[55]:


flights[['DIST', 'AIR_TIME']].corr()


# In[56]:


(flights
    .reset_index(drop=True)
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .pipe(lambda df_:pd.cut(df_.DIST,
          bins=range(0, 2001, 250)))
    .value_counts()
    .sort_index()
)


# In[57]:


zscore = lambda x: (x - x.mean()) / x.std()
short = (flights
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .reset_index(drop=True)    
    .assign(BIN=lambda df_:pd.cut(df_.DIST, bins=range(0, 2001, 250)))
)

scores = (short
    .groupby('BIN')
    ['AIR_TIME']
    .transform(zscore)
)  
(short.assign(SCORE=scores))


# In[58]:


fig, ax = plt.subplots(figsize=(10,6))    
(short.assign(SCORE=scores)
      .pivot(columns='BIN')
      ['SCORE']
      .plot.box(ax=ax)
)
ax.set_title('Z-Scores for Distance Groups')


# In[59]:


mask = (short
    .assign(SCORE=scores)
    .pipe(lambda df_:df_.SCORE.abs() >6)
)


# In[60]:


outliers = (flights
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .reset_index(drop=True)
    [mask]
    .assign(PLOT_NUM=lambda df_:range(1, len(df_)+1))
)


# In[61]:


outliers


# In[62]:


fig, ax = plt.subplots(figsize=(8,6))
(short
    .assign(SCORE=scores)
    .plot.scatter(x='DIST', y='AIR_TIME',
                  alpha=.1, s=1, ax=ax,
                  table=outliers)
)
outliers.plot.scatter(x='DIST', y='AIR_TIME', s=25, ax=ax, grid=True)
outs = outliers[['AIR_TIME', 'DIST', 'PLOT_NUM']]
for t, d, n in outs.itertuples(index=False):
    ax.text(d + 5, t + 5, str(n))
plt.setp(ax.get_xticklabels(), y=.1)
plt.setp(ax.get_xticklines(), visible=False)
ax.set_xlabel('')
ax.set_title('Flight Time vs Distance with Outliers')


# ## 12.6 使用堆疊面積圖找出趨勢

# In[63]:


meetup = pd.read_csv('data/meetup_groups.csv',
                     parse_dates=['join_date'],
                     index_col='join_date')
meetup


# In[64]:


(meetup.groupby([pd.Grouper(freq='W'), 'group']) 
       .size())


# In[65]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
)


# In[66]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
)


# In[67]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
    .pipe(lambda df_: df_.div(
          df_.sum(axis='columns'), axis='index'))
)


# In[68]:


fig, ax = plt.subplots(figsize=(18,6))    
(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
    .pipe(lambda df_: df_.div(
          df_.sum(axis='columns'), axis='index'))
    .plot.area(ax=ax,
          cmap='Greys', xlim=('2013-6', None),
          ylim=(0, 1), legend=False)
)
ax.figure.suptitle('Houston Meetup Groups', size=25)
ax.set_xlabel('')
ax.yaxis.tick_right()
kwargs = {'xycoords':'axes fraction', 'size':15}
ax.annotate(xy=(.1, .7), s='R Users', color='w', **kwargs)
ax.annotate(xy=(.25, .16), s='Data Visualization', color='k', **kwargs)
ax.annotate(xy=(.5, .55), s='Energy Data Science', color='k', **kwargs)
ax.annotate(xy=(.83, .07), s='Data Science', color='k', **kwargs)
ax.annotate(xy=(.86, .78), s='Machine Learning', color='w', **kwargs)


# ## 12.7 了解Seaborn和Pandas之間的區別

# In[69]:


employee = pd.read_csv('data/employee.csv',
                       parse_dates=['HIRE_DATE', 'JOB_DATE'])
employee


# In[70]:


import seaborn as sns


# In[71]:


fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y='DEPARTMENT', data=employee, ax=ax)     


# In[72]:


fig, ax = plt.subplots(figsize=(8, 6))
(employee
    ['DEPARTMENT']
    .value_counts()
    .plot.barh(ax=ax)
)


# In[73]:


fig, ax = plt.subplots(figsize=(8, 6))    
sns.barplot(y='RACE', x='BASE_SALARY', data=employee, ax=ax)


# In[74]:


fig, ax = plt.subplots(figsize=(8, 6))    
(employee
    .groupby('RACE', sort=False) 
    ['BASE_SALARY']
    .mean()
    .plot.barh(rot=0, width=.8, ax=ax)
)
ax.set_xlabel('Mean Salary')


# In[75]:


fig, ax = plt.subplots(figsize=(18, 6))        
sns.barplot(x='RACE', y='BASE_SALARY', hue='GENDER',
            ax=ax, data=employee,
            order=['Hispanic/Latino', 
                   'Black or African American',
                   'American Indian or Alaskan Native',
                   'Asian/Pacific Islander', 'Others',
                   'White'])


# In[76]:


fig, ax = plt.subplots(figsize=(18, 6))            
(employee
    .groupby(['RACE', 'GENDER'], sort=False) 
    ['BASE_SALARY']
    .mean()
    .unstack('GENDER')
    .sort_values('Female')
    .plot.bar(rot=0, ax=ax, width=.8, cmap='viridis')
)


# In[77]:


fig, ax = plt.subplots(figsize=(8, 6))            
sns.boxplot(x='GENDER', y='BASE_SALARY', data=employee,
            hue='RACE', ax=ax)


# In[78]:


fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for g, ax in zip(['Female', 'Male'], axs):
    (employee
        .query('GENDER == @g')
        .assign(RACE=lambda df_:df_.RACE.fillna('NA'))
        .pivot(columns='RACE')
        ['BASE_SALARY']
        .plot.box(ax=ax, rot=30)
    )
    ax.set_title(g + ' Salary')
    ax.set_xlabel('')


# ## 12.8 使用Seaborn進行多變量分析

# In[79]:


emp = pd.read_csv('data/employee.csv',
    parse_dates=['HIRE_DATE', 'JOB_DATE'])

def yrs_exp(df_):
    days_hired = pd.to_datetime('12-1-2016') - df_.HIRE_DATE
    return days_hired.dt.days / 365.25

emp = emp.assign(YEARS_EXPERIENCE=yrs_exp)
emp[['HIRE_DATE', 'YEARS_EXPERIENCE']]


# In[80]:


fig, ax = plt.subplots(figsize=(8, 6))        
sns.regplot(x='YEARS_EXPERIENCE', y='BASE_SALARY', data=emp, ax=ax)


# In[81]:


grid = sns.lmplot(x='YEARS_EXPERIENCE', y='BASE_SALARY', 
                  hue='GENDER',
                  scatter_kws={'s':10}, data=emp)
grid.fig.set_size_inches(8, 6) 


# In[82]:


grid = sns.lmplot(x='YEARS_EXPERIENCE', y='BASE_SALARY',
                  hue='GENDER', col='RACE', col_wrap=3,
                  sharex=False,
                  line_kws = {'linewidth':5},
                  data=emp)
grid.set(ylim=(20000, 120000))     


# In[83]:


deps = emp['DEPARTMENT'].value_counts().index[:2]
deps


# In[84]:


races = emp['RACE'].value_counts().index[:3]
races


# In[85]:


is_dep = emp['DEPARTMENT'].isin(deps)
is_race = emp['RACE'].isin(races)    
emp2 = (emp
    [is_dep & is_race]
    .assign(DEPARTMENT=lambda df_:
            df_['DEPARTMENT'].str.extract('(HPD|HFD)', expand=True))
)
emp2.shape


# In[86]:


emp2['DEPARTMENT'].value_counts()


# In[87]:


emp2['RACE'].value_counts()


# In[88]:


common_depts = (emp.groupby('DEPARTMENT') 
                   .filter(lambda group: len(group) > 50))

fig, ax = plt.subplots(figsize=(8, 6))   
sns.violinplot(x='YEARS_EXPERIENCE', y='GENDER', data=common_depts)


# In[89]:


grid = sns.catplot(x='YEARS_EXPERIENCE', y='GENDER',
                   col='RACE', row='DEPARTMENT',
                   height=3, aspect=2,
                   data=emp2, kind='violin')

