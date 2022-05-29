#!/usr/bin/env python
# coding: utf-8

# # Regular Expression

# * regular expression 使用場景：
#   * text matching: e.g. 大量文本中，matching 出合法的 email 地址
#   * verifying input: e.g. 檢查密碼是否包含大小寫字母與特殊符號
#   * search and replace

# ## regex 101

# * `abc`: `abc`def
# * `123`: 34`123`
# * `\d`, `[0-9]`, `["digit:]`: any digits -> `8`a997b
# * `\D`: any non-digit -> 8`a`997b
# * `\w`, `[A-Za-z0-9_]`: any alpha, numeric
# * `.`: any character -> `a`pple, `p`ig
# * `[abc]`: only a or b or c -> `[cmf]an` => `can`, `man`, `fan`
# * `[^abc]`: not a, b, nor c -> `[^b]og` => `hog`, `dog`, bog
# * `{m}`: m repetitions -> `\d{3}` 連續3個數字; `a{3}` 連續 3 個 a
# * `{m, n}`: 出現 m ~ n 次 -> `\d{3,5}` 連續 3 or 4 or 5 個數字
# * `*`: 0 or more repetitions -> `\a*` 出現 0次以上的 a -> 
# * `?`: 0 or 1 repetition -> `colou?r` -> `color`, `colour`
# * `\s`: 空格 或 tab: `\s+` 一個或多個空白 -> ` abc`, `  abc`
# * `^`: start with
# * `$`: end with
# * `(...)` capture group: `(^file.*)\.pdf$` 表示我要找 file 開頭，任意字0 or 多個，點pdf結尾。那除了會 match 到 `file_record_transcript.pdf` 外，他還抓到 `file_record_transcript` 這個 group
# * `(a(bc))` capture sub-group: e.g. `([:alpha:]{3}\s(\d{4}))` 就是要抓 任意字母3個，空白1個，數字4個，所以可以抓到 `Jan 1987`，然後加上有外和內括號，所以capture到整個group `Jan 1987` 和 sub-group `1987`  
# * `(abc|def)` match abc or def

# ## python 的 `re` module

# In[1]:


import re


# ### re.search

# * 查詢結果，會回傳一個物件。e.g. `res = re.search("ap", "apple)`
# * 它的功能，像是 R 的 
#   * `str_detect` (有沒有 match 到): 這邊用 `res == None` 來檢查，如果是 None，表示沒 match 到。
#   * `str_extract` (抓出 match 到的字串): 可用 `res.span()` 得知 match 到的部分，是在原自串的第幾個位置到第幾個位置; `res.group()` 可以抓出 match 到的字。

# In[42]:


s = "foo123bar"
result = re.search("123", s)
result


# * 可以看到，回傳一個物件，而此物件就有很多 method 和 attribute 可以用

# In[43]:


print(result.span()) # match 的位置是原字串的哪裡到哪裡
print(result.start()) # match 的位置的 start 是哪裡
print(result.end()) # match 的位置的 end 是哪裡
print(result.string) # 要被 match 的字串是哪個


# In[44]:


result = re.search("1234", s) # 如果 match 不到，會回傳 None
result


# In[45]:


result == None


# In[46]:


s = "foo123oof"
re.search("f", s)


# In[47]:


s = "foo123"
re.search("fo*", s) # o* 表示 o 可以重複 0 ~ 多次


# In[48]:


s = "foo123"
re.search("fo+", s) # o+ 表後面 1 個或多個o


# In[49]:


s = "foo"
re.search("o{1,2}", s)


# In[50]:


s = "foo123bar"
print(re.search("\d+",s))
print(re.search("\D+", s))


# In[51]:


s = "one, two, three"
result = re.search("(\w+), (\w+)", s)
result


# In[52]:


print(result.span())
print(result.groups())
print(result.group(0))
print(result.group(1))
print(result.group(2))


# In[53]:


s = "Sep 2010"
result = re.search("(?P<month>\w+)\s(?P<year>\d+)", s) # 幫 group 命名，用 `?P<name><regex>`
print(result)
print(result.groupdict())
print(result.groupdict()["month"])
print(result.groupdict()["year"])


# ### re.match

# * match 和 search 的差別是，他是從第一個字母就開始比對
# * 所以，下面這個例子，他反而 match 不到

# In[40]:


s = "foo123bar"
res = re.match("\d+", s)
res == None


# * `match` 主要用在輸入合法性檢查
# * 例如輸入的自串，是否為 email 形式

# ### re.fullmatch()

# * fullmatch 就是完全 match
# * 例如，要搜尋全都是數字的字串
#   * 用 `re.search` 會寫成 `re.search(r"^\d+$")` ，表示數字開頭，1個數字以上都納入，數字結尾。
#   * 用 `re.fullmatch` 就只要寫成 `re.fullmatch(r"\d+")` 就 ok 了
# * 看例子：

# In[57]:


print(re.fullmatch(r"\d+", "123foo"))
print(re.fullmatch(r"\d+", "foo123"))
print(re.fullmatch(r"\d+", "123"))

print(re.search(r"^\d+$", "123foo"))
print(re.search(r"^\d+$", "foo123"))
print(re.search(r"^\d+$", "123"))


# ### `re.findall()` returns a list of all matches of a regex in a string 

# In[58]:


s = "123foo456bar"
print(re.match("\d+", s))
print(re.search("\d+", s))
print(re.fullmatch("\d+", s))
print(re.findall("\d+", s))


# * 可以清楚看到:
#   * 用之前學的 search, match，都是找到第一組符合的就停止
#   * fullmatch要全都符合才算，所以找不到。
#   * 而 findall 是會繼續往下找，並回給你 list

# ### re.findall + group

# * 這邊澆了這種用法，可以學一下：

# In[59]:


s = "tom,hanks,aaron,tsai,ian,chen"
re.findall(r"(\w+),(\w+)", s)


# * 可以看到，我們用 findall，來讓搜尋可以一路查到底
# * 然後 regex 用 `\w+`，表示1個以上的 alphanumeric 都可。中間用逗點隔開，所以我想找 `alphanumeric, alphanumeric` 這種 pattern
# * 最後，我加了兩個括號，所以，他會幫我抓出兩個 group

# ## python 的逃脫字元 與 raw string

# * 直接講結論，在搜尋時，用 `r""` 來寫，就可以維持和 regex 一致的寫法
# * 例如下例：

# In[62]:


s = "abc\def"
result = re.search(r"(\w+)\\(\w+)", s)
print(result)
print(result.groups())


# * 可以看到，regex，要搜尋特殊字元 `\` 時，前面要再加上逃脫字元 `\`，所以會寫成 `\\`
# * 那我們在搜尋時，在字串前面加上 `r` ，是指 raw index，就可以讓我用 regex 的邏輯去做事
# * 如果沒加 `r`，就會出現 error

# In[65]:


s = "abc\def"
result = re.search("(\w+)\\(\w+)", s)


# * 因為，在 python 中，還要再加上兩個 `\` 來當逃脫字元 (怪...)
# * 所以，要寫成這樣才會對：

# In[67]:


s = "abc\def"
result = re.search("(\w+)\\\\(\w+)", s)
print(result)
print(result.groups())


# * 所以，只要記得，寫 regex 時，都在最前面加上 `r""`，就可以用與 regex 一致的寫法了

# ## 查找和替換

# * 之前在學 string 時，有學過 `.replace()` 這個 method
#   * 例如： `s = "hank_lee"`, `s.replace("_", "~")`  
# * 但這個 replace，只能做到完全 match，如果，我想用 regex 來做替換，那就要學這章教的 (補充： R 的話都統一在一起了，str_replace 時，後面可以直接指定要用 regex 來找)

# ### `re.sub()` 找到 match 的，幫你替換完，再回傳

# In[69]:


s = "foo.123.bar.789,baz"
re.sub(r"\d+", "#", s)


# In[70]:


re.sub(r"[a-z]+", "*", s)


# ### re.subn() 做完 re.sub 的侍候，再回傳給你它取代了幾筆

# In[71]:


re.subn(r'[a-z]+', '*', s)


# * 可以看到，他最後回傳，取代了 3 筆

# ### 加上 group

# * 我如果想把 `123abc456` 這個字串，轉成 `456abc123` 的話，要怎麼做？  
#   * 先用 group，抓出第一組123, 第二組456
#   * 然後 replace 成：`第二組abc第一組`，就搞定了：

# In[73]:


s = "123abc456"
result = re.search(r"(\d+)[a-z]+(\d+)", s)
result.groups()


# In[74]:


re.sub(r"(\d+)[a-z]+(\d+)", r"\2abc\1", s)


# * 上面的 `\2` 就是指第二個group, `\1` 指第一個 group

# ### 加上 function

# * 如果，我想把 `foo.10.bar.20.baz.30` 改成 `FOO.100.BAR.200.BAZ.300`
# * 也就是，我想把英文都改大寫，數字都乘以10。那，怎麼做？
# * 這時就可以用 function

# In[77]:


def foo(match_object):
    s = match_object.group(0)
    if s.isdigit():
        return str(int(s)*10)
    else:
        return s.upper()

s = "foo.10.bar.20.baz.30"
re.sub(r"\w+", foo, s)


# ## 字串分割

# * 基本的做法：

# In[3]:


s = "foo, bar, baz,  qux, quux,     corge"
s.split(", ")


# * 可以看到，因為每個字在逗點後的空格數不同，所以導致切完後，每個字前面還會有不等長的空格
# * 這就要再做一次處理：

# In[4]:



[a.strip() for a in s.split(",")]


# * 那如果用 regex，就不用這麼麻煩了：

# In[6]:


re.split(f",\s*", s) # 切割的 pattern，是逗點後接 0 個以上的空格


# ## flags 的使用

# * 剛剛學過的 `re.search`, `re.match`, `re.xxx` 裡面其實都還有一個參數 `flag = ` 我們沒用過
# * 現在就來介紹這個參數可以怎麼用

# ### `flags = re.IGNORECASE` 可忽略大小寫

# In[8]:


s = "aaaAAA"
print(re.search(pattern = r"a+", string = s))
print(re.search(pattern = r"A+", string = s))
print(re.search(pattern = r"a+", string = s, flags=re.IGNORECASE))


# ### `flags = re.MULTILINE` 會把換行後的字串當新的一行

# * 如下例，我的原始字串雖然有換行符號，但預設下，他還是只當成是一行
# * 這會造成，`bar` 是第二行的起頭，但我用 `^bar` 會抓不到他：

# In[11]:


s = "foo\nbar\nbaz"
print(re.search("^foo", s))
print(re.search("^bar", s))


# * 如果有加 `tags = re.MULTILINE`，就可以解決這個問題，因為他把換行後的字串當新的一行

# In[12]:



print(re.search("^bar", s, re.MULTILINE))


# ### `re.VERBOSE` 可以處理 多行+註解 的 regex

# * 有時候，regex會寫得很長，如果不加註解，也不換行，會非常難閱讀
# * 所以，regex 有可能寫成下面這樣：

# In[15]:


regex = r'''
    (\(\d{2}\))? # (2個數字) 有或沒有都可以, 其實就是要抓區域碼，e.g. (02)
    \s* # 0 個以上的空格
    \d{4} # 3 個數字
    [-.] # - 或 .
    \d{4} # 4 個數字
    $
'''


# In[17]:


print(re.search(regex, "2763-5735"))
print(re.search(regex, "(02) 2763-5735"))


# * 可以發現，都抓不到
# * 但如果加上 `flags = re.VERBOSE` 就 ok 了

# In[19]:


print(re.search(regex, "2763-5735", flags = re.VERBOSE))
print(re.search(regex, "(02) 2763-5735", flags = re.VERBOSE))


# ## 用 `re.compile` 把 regex 變成 object

# * 有時候，我們同一個 regex，會在程式中多次被使用
# * 那如果我把它變成一個 object，那之後有需要修改時，我就只要改一開始定義好的 object 就 ok ，不用到程式裡面去一個一個修改 regex 字串
# * 作法如下：

# In[20]:


re_obj = re.compile(r"ba[rz]", flags = re.IGNORECASE) # 直接定義好，我的 regex 要找 bar 或 baz，然後忽略大小寫
print(re.search(re_obj, "FOOBARBAZ")) # 可以像這樣搜尋
print(re_obj.search("FOOBARBAZ")) # 也可以用 oo 的方式搜尋


# ## Exercise

# * 假設我們拿到以下字串：

# In[21]:


s = """
interface Vlan8
 ip address 192.168.3.50 255.255.255.240
 no ip redirects
 no ip unreachables
 no ip proxy-arp
!
interface Vlan9
 ip address 192.168.3.66 255.255.255.240
 no ip redirects
 no ip unreachables
 no ip proxy-arp
!
interface Vlan10
 ip address 192.168.3.82 255.255.255.240
 no ip redirects
 no ip unreachables
 no ip proxy-arp
!
interface Vlan25
 ip address 192.168.3.211 255.255.255.240
 no ip redirects
 no ip unreachables
 no ip proxy-arp
!
interface Vlan26
 ip address 192.168.3.227 255.255.255.240
 no ip redirects
 no ip unreachables
 no ip proxy-arp
!
interface Vlan99
 bandwidth 10000000
 ip address 192.168.1.2 255.255.255.252
 no ip redirects
 no ip unreachables
 no ip proxy-arp
!
interface Vlan100
 ip address 192.168.192.2 255.255.255.248
 no ip redirects
 no ip unreachables
 no ip proxy-arp
"""


# * 然後，我想找出：
#   * interface name (e.g. Vlan8, Vlan9, Vlan10)
#   * ip (e.g. 255.255.155.140, ...)
#   * mask (e.g. 192.168.3.50)
# * 並把結果寫成以下 dictionary

# In[22]:


result = [
    {'name': 'Vlan8', 'ip': '192.168.3.50', 'mask': '255.255.255.240'},
    {'name': 'Vlan9', 'ip': '192.168.3.66', 'mask': '255.255.255.240'},
    {'name': 'Vlan10', 'ip': '192.168.3.82', 'mask': '255.255.255.240'},
    {'name': 'Vlan25', 'ip': '192.168.3.211', 'mask': '255.255.255.240'},
    {'name': 'Vlan26', 'ip': '192.168.3.227', 'mask': '255.255.255.240'},
    {'name': 'Vlan99', 'ip': '192.168.1.2', 'mask': '255.255.255.252'},
    {'name': 'Vlan100', 'ip': '192.168.192.2', 'mask': '255.255.255.248'}
]


# * 首先來看，我想找 interface name，可以這樣做：

# In[27]:


res1 = re.findall(r"^interface (?P<name>\w+)", s, flags = re.MULTILINE)
res1


# * 找 ip 可以這樣做

# In[29]:


res2 = re.findall(r"ip address (?P<ip>\d+\.\d+\.\d+\.\d+)", s, flags = re.MULTILINE)
res2


# * 找 mask 可以這樣做

# In[31]:


res3 = re.findall(r"ip address [\d\.]+\s(?P<mask>\d+\.\d+\.\d+\.\d+)", sm)
res3


# * 組起來

# In[32]:


result = []
for name, ip, mask in zip(res1, res2, res3):
    temp_dict = dict()
    temp_dict["name"] = name
    temp_dict["ip"] = ip
    temp_dict["mask"] = mask
    result.append(temp_dict)
result


# * 除了上面這種做法外，有另一個更快的，可以這樣做：

# In[34]:


interface_descriptions = re.finditer(
    r"^interface (?P<name>\w+)\n"
    r"( .*\n)*"
    r" ip address (?P<ip>\S+) (?P<mask>\S+)\n",
    s,
    re.MULTILINE)

for part in interface_descriptions:
    print(part.groupdict())


# In[ ]:




