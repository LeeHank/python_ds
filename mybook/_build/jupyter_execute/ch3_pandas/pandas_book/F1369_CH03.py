#!/usr/bin/env python
# coding: utf-8

# # 第3章：建立與保存DataFrame

# ## 3.1 從無到有建立DataFrame

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)

fname = ['Paul', 'John', 'Richard', 'George']
lname = ['McCartney', 'Lennon', 'Starkey', 'Harrison']
birth = [1942, 1940, 1940, 1943]


# In[2]:


people = {'first': fname, 'last': lname, 'birth': birth}


# In[3]:


beatles = pd.DataFrame(people)
beatles


# In[4]:


beatles.index


# In[5]:


pd.DataFrame(people, index=['a', 'b', 'c', 'd'])


# In[6]:


pd.DataFrame([{"first":"Paul","last":"McCartney", "birth":1942},
              {"first":"John","last":"Lennon", "birth":1940},
              {"first":"Richard","last":"Starkey", "birth":1940},
              {"first":"George","last":"Harrison", "birth":1943}])


# In[7]:


pd.DataFrame([{"first":"Paul","last":"McCartney", "birth":1942},
              {"first":"John","last":"Lennon", "birth":1940},
              {"first":"Richard","last":"Starkey", "birth":1940},
              {"first":"George","last":"Harrison", "birth":1943}],
              columns=['last', 'first', 'birth'])


# ## 3.2 存取CSV檔案

# In[8]:


beatles


# In[9]:


from io import StringIO
fout = StringIO()
beatles.to_csv(fout)  


# In[10]:


print(fout.getvalue())


# In[11]:


fout.seek(0)
pd.read_csv(fout)


# In[12]:


_ = fout.seek(0)
pd.read_csv(fout, index_col=0)


# In[13]:


fout = StringIO()
beatles.to_csv(fout, index=False) 
print(fout.getvalue())


# ## 3.3 讀取大型的CSV檔案

# In[14]:


diamonds = pd.read_csv('data/diamonds.csv', nrows=1000)
diamonds


# In[15]:


diamonds.info()


# In[16]:


diamonds2 = pd.read_csv('data/diamonds.csv', nrows=1000,
                        dtype={'carat': np.float32, 'depth': np.float32,
                               'table': np.float32, 'x': np.float32,
                               'y': np.float32, 'z': np.float32,
                               'price': np.int16})

diamonds2.info()


# In[17]:


diamonds.describe()


# In[18]:


diamonds2.describe()


# In[19]:


diamonds2.cut.value_counts()


# In[20]:


diamonds2.color.value_counts()


# In[21]:


diamonds2.clarity.value_counts()


# In[22]:


diamonds3 = pd.read_csv('data/diamonds.csv', nrows=1000,
                        dtype={'carat': np.float32, 'depth': np.float32,
                               'table': np.float32, 'x': np.float32,
                               'y': np.float32, 'z': np.float32,
                               'price': np.int16,
                               'cut': 'category', 'color': 'category',
                               'clarity': 'category'})

diamonds3.info()


# In[23]:


cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']
diamonds4 = pd.read_csv('data/diamonds.csv', nrows=1000,
                        dtype={'carat': np.float32, 'depth': np.float32,
                               'table': np.float32, 'price': np.int16,
                               'cut': 'category', 'color': 'category',
                               'clarity': 'category'},
                        usecols=cols)

diamonds4.info()


# In[24]:


cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']
diamonds_iter = pd.read_csv('data/diamonds.csv', nrows=1000,
                            dtype={'carat': np.float32, 'depth': np.float32,
                                   'table': np.float32, 'price': np.int16,
                                   'cut': 'category', 'color': 'category',
                                   'clarity': 'category'},
                            usecols=cols,
                            chunksize=200)

def process(df):
    return f'processed {df.size} items'

for chunk in diamonds_iter:
    process(chunk)


# In[25]:


np.iinfo(np.int8)


# ### 小編補充

# In[26]:


diamonds4['price'].min()


# In[27]:


diamonds4['price'].max()


# In[28]:


np.finfo(np.float16)


# In[29]:


diamonds.price.memory_usage()


# In[30]:


diamonds.price.memory_usage(index=False)


# In[31]:


diamonds.cut.memory_usage(deep=True)


# In[32]:


get_ipython().system('pip install pyarrow')


# In[33]:


diamonds4.to_feather('d.arr')
diamonds5 = pd.read_feather('d.arr')


# In[34]:


diamonds4.to_parquet('d.pqt')


# ## 3.4 使用Excel檔案

# In[35]:


get_ipython().system('pip install xlwt')


# In[36]:


get_ipython().system('pip install openpyxl')


# In[37]:


get_ipython().system('pip install xlrd')


# In[38]:


beatles.to_excel('beat.xls')


# In[39]:


beatles.to_excel('beat.xlsx')


# In[40]:


beat2 = pd.read_excel('beat.xls')
beat2


# In[41]:


beat2 = pd.read_excel('beat.xls', index_col=0)
beat2


# In[42]:


beat2.dtypes


# In[43]:


xl_writer = pd.ExcelWriter('beat.xlsx')
beatles.to_excel(xl_writer, sheet_name='All')
beatles[beatles.birth < 1941].to_excel(xl_writer, sheet_name='1940')
xl_writer.save()


# ## 3.5 讀取ZIP檔案中的資料 

# In[44]:


autos = pd.read_csv('data/vehicles.csv.zip')
autos


# In[45]:


autos.modifiedOn.dtype


# In[46]:


autos.modifiedOn


# In[47]:


pd.to_datetime(autos.modifiedOn)


# In[48]:


autos = pd.read_csv('data/vehicles.csv.zip', parse_dates=['modifiedOn'])  
autos.modifiedOn


# In[49]:


import zipfile


# In[50]:


with zipfile.ZipFile('data/kaggle-survey-2018.zip') as z:
    print('\n'.join(z.namelist()))
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    kag_questions = kag.iloc[0]
    survey = kag.iloc[1:]


# In[51]:


survey.head(2).T


# ## 3.6 存取資料庫

# In[52]:


import sqlite3
con = sqlite3.connect('data/beat.db')
with con:
    cur = con.cursor()
    cur.execute("""DROP TABLE Band""")
    cur.execute("""CREATE TABLE Band(id INTEGER PRIMARY KEY,
                   fname TEXT, lname TEXT, birthyear INT)""")
    cur.execute("""INSERT INTO Band VALUES(
                   0, 'Paul', 'McCartney', 1942)""")
    cur.execute("""INSERT INTO Band VALUES(
                   1, 'John', 'Lennon', 1940)""")
    _ = con.commit()


# In[53]:


get_ipython().system('pip install sqlalchemy')


# In[54]:


import sqlalchemy as sa
engine = sa.create_engine('sqlite:///data/beat.db', echo=True)
sa_connection = engine.connect()

beat = pd.read_sql('Band', sa_connection, index_col='id')
beat


# In[55]:


sql = '''SELECT fname, birthyear from Band'''
fnames = pd.read_sql(sql, con)
fnames


# ## 3.7 存取JSON格式的資料

# In[56]:


import json
encoded = json.dumps(people)
encoded


# In[57]:


json.loads(encoded)


# In[58]:


beatles = pd.read_json(encoded)
beatles


# In[59]:


records = beatles.to_json(orient='records')
records


# In[60]:


pd.read_json(records, orient='records')


# In[61]:


split = beatles.to_json(orient='split')
split


# In[62]:


pd.read_json(split, orient='split')


# In[63]:


index = beatles.to_json(orient='index')
index


# In[64]:


pd.read_json(index, orient='index')


# In[65]:


values = beatles.to_json(orient='values')
values


# In[66]:


pd.read_json(values, orient='values')


# In[67]:


(pd.read_json(values, orient='values')
   .rename(columns=dict(enumerate(['first', 'last', 'birth'])))
)


# In[68]:


table = beatles.to_json(orient='table')
table


# In[69]:


pd.read_json(table, orient='table')


# In[70]:


output = beat.to_dict()
output


# In[71]:


output['version'] = '0.4.1'
json.dumps(output)


# ## 3.8 讀取HTML表格

# In[72]:


get_ipython().system('pip install lxml')


# In[73]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url)
len(dfs)


# In[74]:


dfs[0]


# In[75]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url, match='List of studio albums', na_values='—')
len(dfs)


# In[76]:


dfs[0].columns


# In[77]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url, match='List of studio albums', na_values='—',
                   header=[0,1])
len(dfs)


# In[78]:


dfs[0]


# In[79]:


df = dfs[0]
df.columns = ['Title', 'Release', 'UK', 'AUS', 'CAN', 'FRA', 'GER',
              'NOR', 'US', 'Certifications']
df


# In[80]:


get_ipython().system('pip install html5lib')


# In[81]:


url = 'https://github.com/mattharrison/datasets/blob/master/data/anscombes.csv'
dfs = pd.read_html(url, attrs={'class': 'csv-data'})
len(dfs)


# In[82]:


dfs[0]

