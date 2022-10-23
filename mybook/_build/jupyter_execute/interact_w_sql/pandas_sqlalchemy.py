#!/usr/bin/env python
# coding: utf-8

# # Pandas + SQLAlchemy

# * 這邊先講結論
# * 連線
#   * 一率用 sqlalchemy 的 `create_engine` 來定義連線 engine，各種資料庫都是 (mysql, mssql, ...)。例如： `engine = create_engine("mysql+pymysql://root:my-secret-pw@localhost:3307/mydev")`
#   * 用 conn 來建立連線，例如 `conn = engine.connect()`
# * 可以直接用 raw sql 的語句，但記得在前面加上 sqlalchemy.sql 裡的 text()，才不會讓 sql 的特殊符號被影響到。例如 `sql_string = "select * from table_A where col_1 like 'C%';"`，那 `%` 在 sql alchemy 中會被當作特殊字元，且沒有逃脫符號可以幫你忙，所以正確做法是這樣： `sql_string2 = text(sql_string)`，再拿 sql_string2 去做事。
# * raw sql 的寫法，就直接用在 CRUD 中，例如：
#   * "改", "刪"，例如 `conn.execute("update table_A set col_1 = 17;")`
#   * "讀"，不會用 `conn.execute(sql_string2)` 的作法，因為他回傳的不是我想要的 pandas dataframe。會用 `pd.read_sql(sql_string2, engine)` 這種方式來做。得到的就會是 pd.DataFrame。但不變的是，我還是用 raw sql 語句，只是丟給不同 function 來做而已。
#   * CRUD 裡面的 "增"，你可以用懶人法 `pd.DataFrame.to_sql()`直接傳上去，缺點就是沒有 schema，所以他自動幫你生的 schema 就是文字和數字而已，沒有 datetime。如果要更細緻的做，就直接用 raw sql 語句，做法就是和剛剛一樣： `conn.execute(sql_string)`

# ## packages

# In[1]:


from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pymysql
import pandas as pd


# ## 連線

# In[2]:


engine = create_engine("mysql+pymysql://root:my-secret-pw@localhost:3307/mydev")
conn = engine.connect()


# ## [增]

# ### 用 `pd.DataFrame.to_sql()`

# * 這是最簡單的方式，也是之前在 R 的時候使用的方式
# * 簡單來說，就是不管 sql table schema，全都用 文字, 數字 的形式來存。犧牲 datetime 的格式。
# * 有關時間的操作，都是從 sql 拉到 local 變成 dataframe 後，再自己轉成時間格式來處理
# * 之後可以再查查看，應該有機會也可以定義 時間, ENUM, SET 等格式。

# In[3]:


user2 = pd.DataFrame({
    "username": ["hank","mike","pink","shark"],
    "height": [185, 172, 153, 198],
    "blood_type":["O", "A", "B", "O"]
})
user2


# In[4]:


user2.to_sql(
    name = "user2", # table name
    con = engine,
    if_exists = "replace", # 'fail', 'replace' 就是 overwrite, 'append'
    index = False
)


# In[5]:


pd.read_sql("select * from user2", engine)


# In[6]:


new = pd.DataFrame({
    "username": ["pinpin","mimi"],
    "height": [160, 170],
    "blood_type": ["B", "B"]
})
new.to_sql("user2", engine, if_exists="append", index = False)


# In[7]:


pd.read_sql("select * from user2", engine)


# ### 用 sql 語句來做

# * 這種作法對熟悉 sql 的人來說蠻直覺的，因為就完全使用 sql 的語句來做

# In[8]:


sql_string = text('''
CREATE TABLE IF NOT EXISTS employee(
  id INT NOT NULL AUTO_INCREMENT,
  first_name VARCHAR(100) NOT NULL,
  last_name VARCHAR(100) NOT NULL,
  title VARCHAR(100) DEFAULT NULL,
  salary DOUBLE DEFAULT NULL,
  hire_date DATE NOT NULL,
  notes TEXT,
  PRIMARY KEY (id)
);
''')
conn.execute(sql_string)


# In[9]:


pd.read_sql("select * from employee", engine)


# In[10]:


sql_string = text('''
INSERT INTO employee (first_name, last_name, title, salary, hire_date) VALUES 
    ('Robin', 'Jackman', 'Software Engineer', 5500, '2001-10-12'),
    ('Taylor', 'Edward', 'Software Architect', 7200, '2002-09-21'),
    ('Vivian', 'Dickens', 'Database Administrator', 6000, '2012-08-29'),
    ('Harry', 'Clifford', 'Database Administrator', 6800, '2015-12-10'),
    ('Eliza', 'Clifford', 'Software Engineer', 4750, '1998-10-19'),
    ('Nancy', 'Newman', 'Software Engineer', 5100, '2007-01-23'),
    ('Melinda', 'Clifford', 'Project Manager', 8500, '2013-10-29'),
    ('Jack', 'Chan', 'Test Engineer', 6500, '2018-09-07'),
    ('Harley', 'Gilbert', 'Software Architect', 8000, '2000-07-17');
''')
conn.execute(sql_string)


# In[11]:


pd.read_sql("select * from employee", engine)


# * 這邊補充一些 sql 語句

# In[12]:


# 寫明 欄位名稱 和 值，如果該表有其他欄位沒被提到，就用 NULL 或 default 帶入
sql_string = text('''
INSERT INTO person(name, phone, age) VALUES ("ABC", "1234", 20);
''')

sql_string = text('''
INSERT INTO person(name, phone, age) VALUES 
    ("ABC", "1234", 20),
    ("DEF", "5678", 21);
''')

# 沒寫明欄位名稱，那就是默認要照該表所有欄位順序來做
sql_string = text('''
INSERT INTO person VALUES ("ABC", "1234", 20);
''')

sql_string = text('''
INSERT INTO person VALUES 
    ("ABC", "1234", 20),
    ("DEF", "5678", 21);
''')


# ## [查] Read

# ### 用 `pd.read_sql()` 查整張表

# In[15]:


pd.read_sql("employee", engine)


# ### 用 sql 語句 (推薦)

# * 一樣，繼續用 `pd.read_sql()` 就好，你丟 sql 字串進去就可以了

# In[16]:


sql_string = text('''
SELECT *
FROM employee
WHERE salary > 5000 and last_name = "Clifford"
''')
pd.read_sql(sql_string, engine)


# * 既然可以下 sql 指令，那各種 order by, filter, group_by, aggregate 的整令，當然就都可以下了

# #### order by

# In[17]:


sql_string = text('''
select * from employee
order by salary;
''')
pd.read_sql(sql_string, engine)


# In[18]:


sql_string = text('''
select * from employee
order by salary desc;
''')
pd.read_sql(sql_string, engine)


# In[19]:


sql_string = text('''
select * from employee
order by salary desc, first_name;
''')
pd.read_sql(sql_string, engine)


# #### limit

# In[20]:


pd.read_sql(text("select * from employee limit 3;"), engine)


# #### like

# * 就是 R 的 `str_detect()`

# In[22]:


# 常用語法
sql_string1 = text("select * from employee where last_name like 'C%';") # C開頭的，%表隨意字元不限字數
sql_string2 = text("select * from employee where last_name like '%i%';") # 中間出現過 i 的
sql_string3 = text("select * from employee where last_name like '%an';") # an 結尾的
sql_string4 = text("select * from employee where last_name like '__an';") # 知道長度是4，結尾是 an
sql_string5 = text("select * from employee where last_name like 'J______';")


# In[23]:


pd.read_sql(sql_string1, engine)


# #### count(*)

# In[36]:


# 整張 table 有幾列
sql_string = text('''
select count(*) from employee;
''')
pd.read_sql(sql_string, engine)


# In[37]:


# title 為 Software Engineer 的table，列數有幾列
sql_string = text('''
select count(*) from employee where title="Software Engineer";
''')
pd.read_sql(sql_string, engine)


# In[38]:


# 各種 title 各有幾列
sql_string = text('''
select title, count(*) from employee
group by title;
''')
pd.read_sql(sql_string, engine)


# #### distinct

# In[39]:


# 有哪些 title?
sql_string = text('''
select distinct(title) from employee;
''')
pd.read_sql(sql_string, engine)


# In[40]:


# 有多少種 title
sql_string = text('''
select count(distinct(title)) from employee;
''')
pd.read_sql(sql_string, engine)


# #### min, max, sum, avg

# In[41]:


# 平均薪水, 最低薪水, 最高薪水, 薪水總合 各是多少
sql_string = text('''
select min(salary), max(salary), sum(salary), avg(salary) from employee;
''')
pd.read_sql(sql_string, engine)


# In[43]:


# 換個名稱比較好讀
sql_string = text('''
select min(salary) as min_salary, 
    max(salary) as max_salary, 
    sum(salary) as sum_salary, 
    avg(salary) as avg_salary
from employee;
''')
pd.read_sql(sql_string, engine)


# In[46]:


# 各種 title 的薪水比較
sql_string = text('''
select title,
    count(*) as n,
    min(salary) as min_salary, 
    max(salary) as max_salary, 
    sum(salary) as sum_salary, 
    avg(salary) as avg_salary
from employee
group by title
order by avg_salary desc;
''')
pd.read_sql(sql_string, engine)


# #### having

# * 這邊只有一個觀念：想在 group_by+summarise 後，想對結果的資料表做篩選，那要用 having，不能用 where
# * 所以，having就是 group by 後，拿來做篩選用的 where。
# * 結論：
#   * 要這樣寫：  `select title, count(*) as n from employee group by title having n>1;`
#   * 不能這樣寫： `select title, count(*) as n from employee group by title where n>1;`

# In[47]:


sql_string = text('''
select title, count(*) as n from employee group by title having n>1;
''')
pd.read_sql(sql_string, engine)


# #### logic operators

# ![image.png](attachment:e95dd848-67c6-4fbb-8e0a-7ecf281ebdcb.png)

# ## [改] Update

# * 因為 pandas 只有做 讀 和 寫 的 wrapper，分別是 `pd.DataFrame.read_sql()` 和 `pd.DataFrame.to_sql()`
# * 對於 update 和 delete，之前在 R 裡面是用 `dbSendUpdate(sql)`，就是直接執行 sql 指令
# * 在 python，就要先建立連線 `conn = engine.connect()`，建立後，就可以 execute 原生的 sql 語句 `conn.execute(sql_string)` 

# In[24]:


sql_string = text('''
update user2
set username = 'hank2'
where height = 185
''')

conn.execute(sql_string)


# In[25]:


pd.read_sql("select * from user2", engine)


# ## [刪] Delete

# * 刪除 table 中的所有列 (但表和 schema 還在)

# In[26]:


sql_string = text('''
delete from user2
''')

# 刪除所有列
# delete from user2

# 刪除某些列
# delete from user2 where xxx

conn.execute(sql_string)


# In[27]:


pd.read_sql("select * from user2", engine)


# * 刪除整張表

# In[28]:


sql_string = text('''
drop table user2
''')

conn.execute(sql_string)


# In[29]:


# pd.read_sql("select * from user2", engine)


# In[ ]:




