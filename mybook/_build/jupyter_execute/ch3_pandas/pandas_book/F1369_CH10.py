#!/usr/bin/env python
# coding: utf-8

# # 第10章：將資料重塑成整齊的形式

# ## 10.1 使用stack()整理『欄位名稱為變數值』的資料

# In[82]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)

state_fruit = pd.read_csv('data/state_fruit.csv', index_col=0)
state_fruit


# In[83]:


state_fruit.stack()


# In[84]:


(state_fruit
   .stack()
   .reset_index()
)


# In[85]:


(state_fruit
   .stack()
   .reset_index()
   .rename(columns={'level_0':'state', 
                    'level_1': 'fruit', 
                    0: 'weight'})
)


# In[86]:


(state_fruit
    .stack()
    .rename_axis(['state', 'fruit'])
)


# In[87]:


(state_fruit
    .stack()
    .rename_axis(['state', 'fruit'])
    .reset_index(name='weight')
)


# In[88]:


state_fruit2 = pd.read_csv('data/state_fruit2.csv')
state_fruit2


# In[89]:


state_fruit2.stack()


# In[90]:


state_fruit2.set_index('State').stack()


# ## 10.2 使用melt()整理欄位名稱為變數值的資料

# In[91]:


state_fruit2 = pd.read_csv('data/state_fruit2.csv')
state_fruit2


# In[92]:


state_fruit2.melt(id_vars=['State'],
                  value_vars=['Apple', 'Orange', 'Banana'])


# In[93]:


state_fruit2.melt(id_vars=['State'],
                  value_vars=['Apple', 'Orange', 'Banana'],
                  var_name='Fruit',
                  value_name='Weight')


# In[94]:


state_fruit2.melt()


# In[95]:


state_fruit2.melt(id_vars='State')


# ## 10.3 同時堆疊多組變數

# In[96]:


movie = pd.read_csv('data/movie.csv')
actor = movie[['movie_title', 'actor_1_name',
               'actor_2_name', 'actor_3_name',
               'actor_1_facebook_likes',
               'actor_2_facebook_likes',
               'actor_3_facebook_likes']]
actor.head()


# In[97]:


def change_col_name(col_name):
    col_name = col_name.replace('_name', '')
    if 'facebook' in col_name:
        fb_idx = col_name.find('facebook')
        col_name = (col_name[:5] + col_name[fb_idx - 1:] 
               + col_name[5:fb_idx-1])
    return col_name


# In[98]:


actor2 = actor.rename(columns=change_col_name)
actor2


# In[99]:


stubs = ['actor', 'actor_facebook_likes']
actor2_tidy = pd.wide_to_long(actor2,
    stubnames=stubs,
    i=['movie_title'],
    j='actor_num',
    sep='_')
actor2_tidy.head()


# In[100]:


df = pd.read_csv('data/stackme.csv')
df


# In[101]:


df.rename(columns = {'a1':'group1_a1', 'b2':'group1_b2',
                     'd':'group2_a1', 'e':'group2_b2'})


# In[102]:


pd.wide_to_long(
    df.rename(columns = {'a1':'group1_a1', 
              'b2':'group1_b2',
              'd':'group2_a1', 'e':'group2_b2'}),
    stubnames=['group1', 'group2'],
    i=['State', 'Country', 'Test'],
    j='Label',
    suffix='.+',
    sep='_')


# ## 10.4 欄位堆疊的反向操作

# In[103]:


usecol_func = lambda x: 'UGDS_' in x or x == 'INSTNM'
college = pd.read_csv('data/college.csv',
                      index_col='INSTNM',
                      usecols=usecol_func)
college


# In[104]:


college_stacked = college.stack()
college_stacked


# In[105]:


college_stacked.unstack()


# In[106]:


college2 = pd.read_csv('data/college.csv', usecols=usecol_func)
college2


# In[107]:


college_melted = college2.melt(id_vars='INSTNM',
                               var_name='Race',
                               value_name='Percentage')
college_melted


# In[108]:


melted_inv = college_melted.pivot(index='INSTNM',
                                  columns='Race',
                                  values='Percentage')
melted_inv


# In[109]:


college2_replication = (melted_inv.loc[college2['INSTNM'], 
                                       college2.columns[1:]]
                                  .reset_index())
college2.equals(college2_replication)


# In[110]:


college.stack().unstack(level=0)


# In[111]:


college.T


# In[112]:


college.transpose()


# ## 10.5 在彙總資料後進行反堆疊操作

# In[113]:


employee = pd.read_csv('data/employee.csv')
(employee
    .groupby('RACE')
    ['BASE_SALARY']
    .mean()
    .astype(int)
)


# In[114]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
)


# In[115]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
    .unstack('GENDER')
)


# In[116]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
    .unstack('RACE')
)


# In[117]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY']
    .agg(['mean', 'max', 'min'])
    .astype(int)
)


# In[118]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY']
    .agg(['mean', 'max', 'min'])
    .astype(int)
    .unstack('GENDER')
)


# ## 10.6 使用groupby模擬pivot_table()的功能

# In[119]:


flights = pd.read_csv('data/flights.csv')
fpt = flights.pivot_table(index='AIRLINE',
    columns='ORG_AIR',
    values='CANCELLED',
    aggfunc='sum',
    fill_value=0).round(2)
fpt


# In[120]:


(flights
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['CANCELLED']
    .sum()
)


# In[121]:


fpg = (flights
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['CANCELLED']
    .sum()
    .unstack('ORG_AIR', fill_value=0)
)


# In[122]:


fpt.equals(fpg)


# In[123]:


flights.pivot_table(index=['AIRLINE', 'MONTH'],
    columns=['ORG_AIR', 'CANCELLED'],
    values=['DEP_DELAY', 'DIST'],
    aggfunc=['sum', 'mean'],
    fill_value=0)


# In[124]:


(flights
    .groupby(['AIRLINE', 'MONTH', 'ORG_AIR', 'CANCELLED']) 
    ['DEP_DELAY', 'DIST'] 
    .agg(['mean', 'sum']) 
    .unstack(['ORG_AIR', 'CANCELLED'], fill_value=0) 
    .swaplevel(0, 1, axis='columns')
)


# ## 10.7 重新命名各軸內的不同層級

# In[125]:


college = pd.read_csv('data/college.csv')
(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
)


# In[126]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
)


# In[127]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
)


# In[128]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .swaplevel('AGG_FUNCS', 'STABBR', axis='index')
)


# In[129]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .swaplevel('AGG_FUNCS', 'STABBR', axis='index') 
    .sort_index(level='RELAFFIL', axis='index') 
    .sort_index(level='AGG_COLS', axis='columns')
)


# In[130]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .unstack(['RELAFFIL', 'STABBR'])
)


# In[131]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack(['AGG_FUNCS', 'AGG_COLS'])
)


# In[132]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .unstack(['STABBR', 'RELAFFIL']) 
)


# In[133]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis([None, None], axis='index') 
    .rename_axis([None, None], axis='columns')
)


# ## 10.8 重塑『欄位名稱包含多個變數』的資料

# In[134]:


weightlifting = pd.read_csv('data/weightlifting_men.csv')
weightlifting


# In[135]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)


# In[136]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
)


# In[137]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
)


# In[138]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
    .assign(Sex=lambda df_: df_.Sex.str[0])
)


# In[139]:


melted = (weightlifting.melt(id_vars='Weight Category',
                             var_name='sex_age',
                             value_name='Qual Total'))
tidy = pd.concat([melted['sex_age'].str.split(expand=True)
                                   .rename(columns={0:'Sex', 1:'Age Group'})
                                   .assign(Sex=lambda df_: df_.Sex.str[0]),
                  melted[['Weight Category', 'Qual Total']]],
                  axis='columns')
tidy


# In[140]:


melted = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)
(melted
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
    .assign(Sex=lambda df_: df_.Sex.str[0],
            Weight_Category=melted['Weight Category'],
            Quad_Total=melted['Qual Total'])
)


# In[141]:


tidy2 = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    .assign(Sex=lambda df_:df_.sex_age.str[0],
            **{'Age Group':(lambda df_: (df_
                .sex_age
                .str.extract(r'(\d{2}[-+](?:\d{2})?)',
                             expand=False)))})
    .drop(columns='sex_age')
)

tidy2


# In[142]:


tidy.sort_index(axis=1).equals(tidy2.sort_index(axis=1))


# ### 小編補充

# In[143]:


import re

match = re.search(r'(\d{2})-(\d{2})','M35 35-39')
print(match.group(0))
print(match.group(1))
print(match.group(2))


# ## 10.9 重塑『多個變數儲存在單一欄位內』的資料

# In[144]:


inspections = pd.read_csv('data/restaurant_inspections.csv',
                          parse_dates=['Date'])
inspections


# In[145]:


# inspections.pivot(index=['Name', 'Date'],
#                   columns='Info', 
#                   values='Value')


# In[146]:


inspections.set_index(['Name','Date', 'Info'])


# In[147]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
)


# In[148]:


(inspections
    .set_index(['Name','Date','Info']) 
    .unstack('Info')
    .reset_index(col_level=-1)
)


# In[149]:


def flatten0(df_):
    df_.columns = df_.columns.droplevel(0).rename(None)
    return df_

(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
    .reset_index(col_level=-1)
    .pipe(flatten0)
)


# In[150]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .squeeze() 
    .unstack('Info') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# In[151]:


(inspections
    .pivot_table(index=['Name', 'Date'],
                 columns='Info',
                 values='Value',
                 aggfunc='first') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ## 10.10 整理『單一儲存格中包含多個值』的資料

# In[152]:


cities = pd.read_csv('data/texas_cities.csv')
cities


# In[153]:


geolocations = cities.Geolocation.str.split(pat='. ', expand=True)
geolocations.columns = ['latitude', 'latitude direction',
                        'longitude', 'longitude direction']


# In[154]:


geolocations = geolocations.astype({'latitude':'float', 'longitude':'float'})
geolocations.dtypes


# In[155]:


geolocations.apply(pd.to_numeric, errors='ignore')


# In[156]:


geolocations.assign(city=cities['City'])


# In[157]:


cities.Geolocation.str.split(pat=r'° |, ', expand=True)


# In[158]:


cities.Geolocation.str.extract(r'([0-9.]+). (N|S), ([0-9.]+). (E|W)', expand=True)


# ## 10.11 整理『欄位名稱及欄位值包含變數』的資料

# In[159]:


sensors = pd.read_csv('data/sensors.csv')
sensors


# In[160]:


sensors.melt(id_vars=['Group', 'Property'], var_name='Year')


# In[161]:


(sensors
    .melt(id_vars=['Group', 'Property'], var_name='Year') 
    .pivot_table(index=['Group', 'Year'],
                 columns='Property', values='value') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# In[162]:


(sensors
    .set_index(['Group', 'Property']) 
    .stack() 
    .unstack('Property') 
    .rename_axis(['Group', 'Year'], axis='index') 
    .rename_axis(None, axis='columns') 
    .reset_index()
)

