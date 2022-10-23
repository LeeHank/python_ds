from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base

# 建立 table schema
BASE = declarative_base()  # BASE 現在是一個 factory class
class User(BASE):

    # 定義 table 名稱 (通常不用寫，因為 class 就代表 table，也就是 User)
    # 這邊只是示範，要定義 table name，也可以這樣定
    __tablename__ = 'user'

    # 定義 User 這張 table，有哪些 column
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    username = sa.Column(sa.String(64))
    password = sa.Column(sa.String(64))
    email = sa.Column(sa.String(128), unique=True)
    create_at = sa.Column(sa.DateTime, server_default=sa.func.now())
    
    def __repr__(self):
        return f"id={self.id}, username={self.username}, email={self.email}"


engine = sa.create_engine("mysql+pymysql://root:my-secret-pw@localhost:3307/demo")
Session = sa.orm.sessionmaker(bind=engine) # 此時的 session 是個函數，不是值
BASE.metadata.create_all(engine) # 會把所有繼承 BASE 的 object (相當於 sql 的 table)，都上傳上去



# [增] 插入列 ----
## 既然 User 這個 class 代表 sql 的 table
## 那 User 所實例化的 instance，就代表該 table 裡的一個 row
user1 = User(username = 'test1', password = 'test1', email = 'test1@test1.com')
user2 = User(username = 'test2', password = 'test2', email = 'test2@test2.com')
user3 = User(username = 'test3', password = 'test3', email = 'test3@test3.com')

## 插入時，要先建立一個 session
session = Session()
## 單筆加入
session.add(user1)

## 多筆加入
session.add_all([user2, user3])

## commit (就和 git commit 一樣，做 commit，才把 session 裡的東西做執行)
session.commit()

# [查]
s = Session()
users = s.query(User)
for user in users:
    print(user)
    
    
    

conn = engine.connect()
res = conn.execute("select * from user")

aa = pd.read_sql_table('user', engine)
sql_string1 = '''
SELECT
    username, password
FROM user
WHERE username = 'test1'
'''
pd.read_sql_query(sql_string1, engine)


aa.create_at.dtype

for i in res:
    i
import pandas as pd
aa = pd.DataFrame(res.fetchall())
aa.columns = res.keys()