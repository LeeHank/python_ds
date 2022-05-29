#!/usr/bin/env python
# coding: utf-8

# # Beginner (Official)

# * NumPy 是 Numerical Python 的縮寫，是一個 open source Python library
# * Numpy 被頻繁地用在 Pandas, SciPy, Matplotlib, scikit-learn, scikit-image ... 等package中
# * Numpy 包含兩種 data structure:  
#   * multidimensional array (i.e. `ndarray`)  
#   * matrix
# * Numpy 有一堆高階數學function，可以對 ndarray 和 matrix 做計算

# In[1]:


import numpy as np


# ## 定義 ndarray

# * 在 numpy 世界中，不是只有 tabular 這種資料型態。  
# * 事實上，他把資料的儲存和計算，generalize 到 ndarray (n維陣列/n階張量) 之中. 
# * 舉例來說：
#   * 我收到 100 個人的身高，我可以儲存成一個 1d-array: (e.g. [172, 189, 164, ..., 158]) -> $R^{100人}$ 
#   * 我收到 100 個人的身高和體重，我可以儲存成 2d-array: (e.g. [[172,...,158],[65,...,54]]) -> $R^{2變數 \times 100人}$
#   * 我收到 1 張 8x8像素的灰階影像的數據，是個矩陣型資料，可以存到 2d-array 中 -> $R^{8像素 \times 8像素}$ 
#   * 1張彩色影像數據，會是RGB 3 個通道，每個通道都是 8x8 像素的圖片，可以存到 3d-array 中  $R^{3通道 \times 8像素 \times 8像素}$
#   * 10 張彩色影像數據，可以存到 4d-array 中  -> $R^{10張 \times 3通道 \times 8像素 \times 8像素}$
#   * 5批資料，每批都有 10 張彩色影像，可以存到 5d-array中 $R^{5 \times 10 \times 3 \times 8 \times 8}$ 
#   * 3個年份，每個年份都有5批資料，每批都有 10 張彩色影像，可以存到 6d-array中 $R^{3年份 \times 5批 \times 3通道 \times 8像素 \times 8像素}$  
# * 所以，這邊先講結論，等等會舉例和示範：  
#   * ndarray，就是 n "層" list 的結構，要用階層的角度去思考. 
#   * ndarray 的 d，是 dim 的縮寫，他的 dim 是指 R 的上面有幾個數字相乘(i.e. $R^3$, $R^2$ 的 R 上面都只有一個數字，所以他是 1d; $R^{2 \times 3}$, $R^{100 \times 5}$ 的 R 上面都是 2 個數字相乘，所以是 2d; 後面以此類推，$R^{3 \times 5 \times 3 \times 8 \times 8}$ 的 R 上面有 6 個數字相乘，所以是 6d
#   * 所以，ndarray的 dim，和線性代數的 dim 不同，因為他的dim是要描述資料儲存需要幾層，但線性代數的dim是在描述要span出一個空間需要幾個基底向量 (所以線代的 $R^3$ 是指 span 出這個空間要3個向量，dim=3; dim($R^4$)=4; dim($R^{2x3}$)=6，不要搞混了)
#   * 最後，繼續混淆你的是，$R^{3x4}$ 他叫 2d-array，表示 dim = 2 (實際上也是用 `.ndim` 這個 attribute去看維度)，但他又喜歡稱呼他為 2-axes(兩個軸)，第一個 axis 的 size = 3, 第二個 axis 的 size = 4  
# * 開始看實例吧：

# ### 1d-array

# * 就是只有 `1層` 的結構

# In[8]:


# 1d-array
a = np.array([1,2,3])
a


# * 因為所有的 element 都是數字，所以我可以做數學計算：

# In[9]:


a.mean()


# ### 2d-array

# * 對於1張灰階影像資料，例如是這樣的一張矩陣型資料: $\left[\begin{array}{cc} 0 & 1\\1 & 0 \\1 & 1\end{array}\right]$，可以用數學寫成： $\boldsymbol{X} \in R^{3 \times 2}$ 
# * 在 python 中，會用這樣的 array 來儲存他：  

# In[10]:


a = [
  [0, 1], 
  [1, 0], 
  [1, 1]
]
a = np.array(a)
a


# * 我們總是會很想用矩陣的角度去看他，但拜託你忍一忍，不要這樣做。因為之後要一路推廣下去。
# * 所以，我們現在改成用層次的方式來理解他：$R^{3 \times 2}$ 就讀成: 總共有3列，每一列都有2筆數據。
# * 那他的階層就會長成：  
#   * 第一列: [0, 1]  
#     * 第一列的第一個 element: 0  
#     * 第一列的第二個 element: 1  
#   * 第二列: [1, 0]  
#     * 第二列的第一個 element: 1  
#     * 第二列的第二個 element: 0  
#   * 第三列: [1, 1]  
#     * 第三列的第一個 element: 1  
#     * 第三列的第二個 element: 1  
# * 也就是第一層是 $R^{3 \times 2}$ 的 3，第二層是 $R^{3 \times 2}$ 的 2
# * 所以，我們要練習，這樣寫 list：  

# In[16]:


# 第一步，先寫出第一層，有3列： 
# a = [[], [], []] 

# 第二步，再把第二層的內容補進去，各2個element： 
a = [[0, 1], [1, 0], [1, 1]]

# 第三步，轉成 np.array
a = np.array(a)


# * 接著，來定義一些名詞： $R^{3 \times 2}$，R的上面有`2`個數字相乘，我們稱它為`2`階張量，儲存的資料類型是 `2`d array。也就是說，這個張量的`維度是2`。然後 R 上面的長相是 $3 \times 2$，所以我們說他的 shape 是 `(3,2)`  
# * 我們來看一下這個 numpy array 的 attribute，就可以驗證上面講的內容：

# In[17]:


a.ndim


# * ndim 是 2，就表示 ndarray 是 2d array(n=2, 有兩層，R上面有2個數字相乘)  

# In[18]:


a.shape


# * shape 是 (3,2)，表示他是 $R^{3 \times 2}$ 的張量  

# ### 3d-array

# In[6]:


# 2d-array
b = np.array([[1,2],[3,4]])
b


# In[5]:


# 2d-array (用 tuple，自動幫你轉回list)
b = np.array([(1,2),(3,4)])
b


# * 對於1張彩色影像資料，他會有3張矩陣型資料，例如長成這樣：  
# 
# $$
# \left[
# \left[\begin{array}{cc} 0 & 1\\1 & 0 \\1 & 1\end{array}\right],
# \left[\begin{array}{cc} 0 & 0\\1 & 1 \\1 & 0\end{array}\right],
# \left[\begin{array}{cc} 1 & 1\\0 & 0 \\0 & 1\end{array}\right]
# \right]
# $$
# 
# * 那我可以寫成這樣：$\boldsymbol{X} \in R^{3 \times 3 \times 2}$  
# 
# $$
# \boldsymbol{X} = \left[
# R^{3 \times 2},
# G^{3 \times 2},
# B^{3 \times 2}
# \right]
# $$
# * 由 $R^{3 \times 3 \times 2}$ 已可知道，他是 3d array(所以要給他3層)。shape是 `3*3*2`，所以第一層有3個 element，第二層有3個element，第三層有2個element。  
# * 那我再造 list 時，第一步就是先寫第一層：  
# 
# ```
# a = [
#   [],
#   [],
#   []
# ]
# ```
# 
# * 然後第二層：  
# 
# ```
# a = [
#   [
#     [],
#     [], 
#     []
#   ],
#   [
#     [],
#     [], 
#     []
#   ],
#   [
#     [],
#     [], 
#     []
#   ]
# ]
# ```
# 
# * 最後，做出第三層：  
# 

# In[19]:


a = [
  [
    [0, 1],
    [1, 0], 
    [1, 1]
  ],
  [
    [0, 0],
    [1, 1], 
    [1, 0]
  ],
  [
    [1, 1],
    [0, 0], 
    [0, 1]
  ]
]
a = np.array(a)
a


# * 驗證一下，這個 $R^{3 \times 3 \times 2}$ 是 3d array(因為R上面有3個數字相乘，或說，建立list的時候要寫到第3層)。shape是 `3*3*2`

# In[20]:


print(f"the dim of a is {a.ndim}")
print(f"the shape of a is {a.shape}")


# ### 4d-array

# * 剛剛介紹完，1張彩色影像資料要如何儲存。那如果 2 張彩色影像數據，要如何存到 list 中？  
# * 很簡單嘛，現在變成是一個 $R^{2張 \times 3通道 \times 3列 \times 2行}$ 的資料，所以我要做一個 4D array(因為 R 上面有4個數字相乘，list要做到4層)，然後他的 shape 會是 `(2,3,3,2)`  
# * 開始造 list ，第一步就是先寫第一層(2張圖片)：  
# 
# ```
# a = [
#   [],
#   []
# ]
# ```
# 
# * 然後第二層，每張圖片，都有RGB三個通道：  
# 
# ```
# a = [
#   [
#     [],
#     [], 
#     []
#   ],
#   [
#     [],
#     [], 
#     []
#   ]
# ]
# ```
# 
# * 然後，第三層，每個 RGB 中，都有三列：  
# 
# ```
# a = [
#   [
#     [
#       [],
#       [],
#       []
#     ],
#     [
#       [],
#       [],
#       []
#     ], 
#     [
#       [],
#       [],
#       []
#     ]
#   ],
#   [
#     [
#       [],
#       [],
#       []
#     ],
#     [
#       [],
#       [],
#       []
#     ], 
#     [
#       [],
#       [],
#       []
#     ]
#   ]
# ]
# ```
# 
# * 最後，每一列裡面，都有兩個 element:  

# In[21]:


a = [
  [
    [
      [0, 1],
      [1, 0], 
      [1, 1]
    ],
    [
      [0, 0],
      [1, 1], 
      [1, 0]
    ], 
    [
      [1, 1],
      [0, 0], 
      [0, 1]
    ]
  ],
  [
    [
      [0, 0],
      [1, 0], 
      [0, 1]
    ],
    [
      [1, 1],
      [1, 1], 
      [1, 1]
    ], 
    [
      [0, 0],
      [0, 1], 
      [1, 0]
    ]
  ]
]
a = np.array(a)
a


# * 驗證一下，這個 $R^{2張 \times 3通道 \times 3列 \times 2行}$是 4d array(因為R上面有4個數字相乘，或說，建立list的時候要寫到第4層)。shape是 `2*3*3*2`

# In[22]:


print(f"the dim of a is {a.ndim}")
print(f"the shape of a is {a.shape}")
print(a.size)
len(a)


# ## dtype

# * numpy的資料類型包括：  
#     * int (帶符號的整數)
#       * int8
#       * int16
#       * int32
#       * int64 (預設)
#     * unit (不帶符號的整數)  
#       * unit8
#       * unit16
#       * unit32
#       * unit64
#     * float (浮點數)
#       * float16  
#       * float32
#       * float64 (預設)
#       * float128
#     * bool

# In[17]:


a = np.array([0, 1, 2])
print(a)
print(a.dtype)


# In[18]:


b = np.array([0, 1, 2], dtype = "float64")
print(b)
print(b.dtype)


# In[20]:


# 轉換 type
c = np.array(b, dtype = "int32")
print(c)
print(c.dtype)


# ## subsetting, slicing, indexing

# * subsetting 就是取特定位置，例如 a[[0,2]]. 
# * slicing 是取起點到終點的位置，例如 a[0:3]  
# * 這邊就一起講吧

# ### 1-d

# ![](figures/np_indexing.png)

# In[21]:


a = np.arange(10) # 0~9 共 10 個元素的 array
a


# In[40]:


# subsetting
print(a[0])
print(a[[0, 2, 4]])


# In[41]:


# slicing
a[1:5] # start = 1, stop = 5(所以不包含5) => 取 index = 1 ~ 4


# In[42]:


a[2:8:2] # start = 2, stop = 8(所以不包含8), step = 2, 所以是取 index = 2, 4, 6


# In[43]:


print(a[0:2])
print(a[:2]) # 意思一樣


# In[44]:


a[-1] # 最後一個


# In[45]:


a[-3] # 從後數來第三個


# In[46]:


print(a[-3:]) # 從後數來第三個，到最後
print(a[-3:-1]) # start = -3, stop = -1 (所以不包含-1)


# In[47]:


print(a[4:]) # 取 index = 4 到最後
print(a[4:-1]) # start = 4, stop = 最後(所以不包含最後)


# In[48]:


a[::-1] # a[::] 表示從最後到最前的全部, step = -1, 所以是整個倒過來的意思


# In[49]:


a[::2] # 每間隔兩個取一次


# In[50]:


# booling indexing
bool_ind = a > 5
bool_ind


# In[51]:


a[bool_ind]


# ### 2-d

# In[54]:


b = np.arange(20).reshape(4,5)
b


# In[55]:


b[1:3, 2:4] # 第一個軸，取 start = 1, stop = 3(不包含3)，所以是取 index = 1, 2 的列，也就是第 2, 3 兩列
            # 第二個軸，取 start = 2, stop = 4(不包含4)，所以是取 index = 2, 3 的行，也就是第 3, 4 兩行


# In[56]:


b[:2, 1:] # index = 0~1 的列 (i.e. 第1, 2 列); index = 1:最後的行 (i.e. 第 2, 3, 4, 5 行)


# In[60]:


# booling indexing
b < 6


# In[61]:


b[b<5]


# ## Broadcasting

# * broadcasting 是 numpy 在背後幫我們做的，那這邊要來了解一下，背後幫我們做了哪些事

# ### 增軸

# In[63]:


a = np.array([[1,2]])
b = np.array([3,4])

print(a.shape)
print(b.shape)


# * 可以看到， a 有兩個軸 (因為 shape 有兩個 element); b 有 1 個軸. 
# * 但，這兩個 array 還是可以相加，這是因為，他會幫 b 多增加一個軸，變成 [[3,4]], shape 為 (1,2)，然後再相加

# In[64]:


a + b


# ### 增維

# * 另外一種，是增加維度。也就是原本大家的軸是一樣的，但維不一樣，例如以下：

# In[73]:


data = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([[1, 1]])

print(data.shape) # 2軸, 維度分別為 3 與 2
print(ones_row.shape) # 2 軸, 維度是 1 與 2

res = data + ones_row
print(res)
print(res.shape)


# ![](figures/np_matrix_broadcasting.png)

# * 增軸 和 增維 可以同時發生，例如底下這個 case

# In[75]:


data = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([1, 1])

print(data.shape)
print(ones_row.shape)

print(data+ones_row)


# * numpy 把 ones_row，先增家了第一個軸，在幫他把維度調成3，才能與 data 相加
# * 同樣的 case 如下：

# In[76]:


a = np.array([1,2])
b = np.array(1.6)

print(a.shape) # 1軸, 維度2
print(b.shape) # 0 軸

print(a*b)
print((a*b).shape)


# ![](figures/np_multiply_broadcasting.png)

# ## 建立特殊 array 

# * 我們可以造出各種特殊的 array
#   * 造出 constant array (e.g. `np.zeros()`, `np.ones()`, `np.full()`  
#   * 做出單位矩陣 (e.g. `np.eye()`)
#   * 做出等距的向量 (e.g. `np.arange()`, `np.linspace()`)
#   * 做出隨機向量 (e.g. `np.random.*()`)

# ### np.arange()

# * 給公差，建立等差數列

# In[27]:


np.arange(10) # 預設公差 = 1


# In[3]:


np.arange(10,25,3) # start = 10, stop = 25(所以不包含25), 公差是 3


# ### np.linspace()

# * 給個數，幫你生均勻數列

# In[4]:


np.linspace(10,25,5) # start = 10, end = 25(所以包含25), 在 10~25 中，幫我生出 5 個數字


# ### np.zeros()

# In[23]:


np.zeros(4)


# In[24]:


np.zeros((3,2))


# ### np.zeros_like()

# * 可以幫你轉成某個 array 的形狀，然後裡面全補 0

# In[124]:


a = np.arange(9).reshape(3,3)
a


# In[125]:


np.zeros_like(a)


# ### np.ones()

# In[25]:


np.ones(4)


# In[26]:


np.ones((3,2))


# ### np.ones_like()

# In[126]:


a = np.arange(9).reshape(3,3)
a


# In[127]:


np.ones_like(a)


# ### np.empty()

# * 如果想，隨便產出一個 array的話，就用他
# * 雖然寫 empty，但其時是隨意給值的意思，看例子就懂了：

# In[128]:


np.empty((3,3))


# * 這個 funtion 的效率比 np.ones() 和 np.zeros() 好很多，所以如果我們想先做出某個 shape 的 array，之後再把正確的值填入的話，用 np.empty() 就對了

# ### np.full()

# In[33]:


np.full(5, 4) # 做一個 shape = 5 的 array，element 都是 4


# In[35]:


np.full((3,2), 4) # 做一個 shape = (3,2) 的 array，element 都是 4


# ### 單位矩陣： np.identity()

# In[129]:


np.identity(3)


# In[130]:


np.identity(5)


# ### 單位矩陣加強版: np.eye()

# * np.eye() 除了可以做到 np.identity() 的事外：

# In[131]:


np.eye(3)


# In[132]:


np.eye(5)


# * np.eye() 還可以做出非方陣：

# In[133]:


np.eye(3,5)


# In[134]:


np.eye(5,3)


# * np.eye() 還有 k 參數，可以讓 1 從對角線，往上k單位，或往下 k 單位

# In[135]:


np.eye(5, k = 1) # 往上 1 單位


# In[136]:


np.eye(5, k = -2) # 往下 2 單位


# ## `np.random` module 

# In[3]:


from numpy import random


# ### distribution

# #### `rand()`: 我想產出 U(0,1) 的亂數

# In[6]:


random.rand(10) # 產出 shape = (10,) 的 U(0,1) 亂數


# In[7]:


random.rand(2,3) # 產出 shape = (2,3) 的 U(0,1) 亂數


# #### 我想產出 U(a,b) 的亂數

# * 這就要自己動點手腳  
# * 如果 $X \sim U(a,b)$，那 $\frac{X-a}{b-a} \sim U(0,1)$
# * 所以： $X \sim (b-a) \times U(0,1) + a$

# In[13]:


(5-3)*random.rand(5) + 3 # 產生 shape = (5,) 的 U(3,5) 亂數


# #### `randint()`: 我想產出 U(low, high) 的整數值亂數

# In[14]:


random.randint(3, 9, 3) # start = 3, stop = 9(不包含9)，產出 shape = (3,) 的 U(3,8) 整數值亂數


# In[15]:


random.randint(3, 9, (3,3)) # start = 3, stop = 9(不包含9)，產出 shape = (3,3) 的 U(3,8) 整數值亂數


# #### 我想產出 N(0,1) 的亂數

# #### 我想產出 N(mean, sigma) 的亂數

# ### 設 seed

# * 在我們產生任何亂數時，他都會用產生亂數的時間，來當 seed，所以每次產出的結果都不會一樣 (因為時間都不同). 
# * 那我們可以在產生亂數之 `前`，先用 `random.seed()` 來設 seed ，就能讓結果相同

# In[17]:


random.seed(3)
print(random.rand(5))

print(random.rand(5)) # 此時的 seed 是用執行這行時的時間當 seed，所以會和第一行結果不同

random.seed(3)
print(random.rand(5)) # seed 設的跟第一次相同，所以結果相同


# ### 隨機抽樣

# In[22]:


a = ["python", "Ruby", "Java", "JavaScript", "PHP"]

random.choice(a, 3) # 隨機抽出 3 個, 預設是取後放回，也就是 replace = True


# In[29]:


random.choice(a, 10, replace = True) # 因為是取後放回，所以可以取超過選單數量的個數


# In[30]:


random.choice(a, 10, replace = True, p = [0.8, 0.05, 0.05, 0.05, 0.05]) # 依不同權重來抽樣


# In[31]:


random.choice(a, 5, replace = False) # 取後不放回，所以最多只能取 5 個


# In[32]:


random.choice(9, 5, replace = False) # 第一個 argument 寫 9，等於 range(9)的意思，所以會從 0~8 做抽樣


# ### 隨機排列

# In[36]:


a = np.arange(10)
a


# In[37]:


random.shuffle(a) # 隨機重排後，存回 a (所以不用再 assign 回 a)
a


# In[8]:


orig = np.arange(9).reshape((3,3))
orig


# In[13]:


random.shuffle(orig) # 會 by column 內, 自己 random


# In[14]:


orig


# ## Sorting

# * np.sort: 排序
# * np.argsort: 回傳 index，照他的 index 取值就可完成排序

# ### 1-d array

# In[137]:


a = np.random.randint(0, 100, size = 20)
a


# In[138]:


np.sort(a)


# In[139]:


ind = np.argsort(a)
ind


# In[140]:


a[ind]


# ### 2-d array

# In[141]:


a = np.random.randint(0, 100, size = 20).reshape(4,5)
a


# #### 各 column 自己排序

# * 如果想在每個 column 裡面做排序，那用 `axis = 0`

# In[142]:


np.sort(a, axis = 0)


# * 如果用 argsort：

# In[144]:


ind = np.argsort(a, axis = 0)
ind


# * 要對每一行來看，例如第一行是 2,3,0,1 ，表示第一行如果要排序的話，要先取 index = 2, 接著 3, 0, 1。其他每一行也都是這樣看

# #### 各 row 自己排序

# * 如果想在每個 row 裡面做排序，那用 `axis = 1`

# In[143]:


np.sort(a, axis = 1)


# * 如果用 argsort

# In[147]:


np.argsort(a, axis = 1)


# * 現在變成要每一列來看。
# * 例如第一列，是 0, 4, 3, 1, 2，表示第一列要排序時，要先取 index = 0, 再來 4, 3, 1, 2

# ### 3-d array

# In[148]:


c = np.random.randint(0, 100, size = (2, 4, 5))
c


# * 如果對 `axis = 0` 做排序，那就是對最外層做排序，所以是對這兩個 2d-array 的對應 element 做比較，然後排大小：

# In[149]:


np.sort(c, axis = 0)


# * 可以看到，現在上下兩個 2d array，每一個 element 都是上面的小，下面的大
# * 用 argsort 也可以看到這個現象：

# In[150]:


np.argsort(c, axis = 0)


# * 看上下兩個 2d array，
#   * 第一個 element 上下對應到的值是 0, 1，表示要先選 index = 0(上面的)，再選 index = 1(下面的)，才完成排序
#   * 第二個 element，上下對應到的值是 1, 0，表示要先選 index = 1(下面的)，再選 index = 0(上面的)，才完成排序

# ### .sort() method

# * 和剛剛的用法完全一樣，差別在，他不會吐值，而是 in-place 取代掉原本的 object

# In[151]:


a = np.random.randint(0, 100, size = 20)
a


# In[152]:


a.sort() # 不會吐值出來，排序完，in-place 取代掉原物件 a


# In[153]:


a


# In[154]:


a = np.random.randint(0, 100, size = 20).reshape(2,10)
a


# In[155]:


a.sort()
a


# In[156]:


a.sort(axis = 0)
a


# ## Array 改形狀

# ### 調 shape: reshape (mutable)/ resize (immutable)

# ![](figures/np_reshape.png)

# #### 指定新 shape 的 np function 一樣

# In[105]:


# 兩種的做法都一樣，元素都是 by row 填入
a = np.arange(6)
print(np.reshape(a, (2,3)))
print(np.resize(a, (2,3)))


# #### reshape 可以是 ndarray 物件的 method，但 resize 不是

# In[104]:


a = np.arange(6)
print(a.reshape(2, 3)) # method 的寫法，ok
print(a.resize(2, 3)) # 無法


# * 如果我寫 -1 ，那 np 會幫我算出適合的數量

# In[88]:


a.reshape(2, -1) # 行數寫 -1 ，他會自動幫你算出，3 比較適合


# In[89]:


a.reshape(-1, 2) # 列數寫 -1，他會自動幫你算出 -3 比較適合


# #### reshape 可以用 -1, resize 不行

# * 在轉shape的時候，如果填 -1 ，他會自動幫你找到對的 colum, row 數量

# In[113]:


a.reshape(2, -1) # 行數寫 -1 ，他會自動幫你算出，3 比較適合


# In[114]:


a.reshape(-1, 2) # 列數寫 -1 ，他會自動幫你算出，3 比較適合


# * 如果是 resize, 他會直接報 error

# In[115]:


np.resize(a, (2, -1))


# In[116]:


a.resize(2, -1)


# #### resize 可轉成不適配的維度, reshape 不行

# In[110]:


a = np.arange(6)
b = np.resize(a, (2, 4))
print(b)


# * 可以看到，他會幫你循環補值。但如果用 reshape，他會直接報 error

# #### reshape 可以 by column 來排， resize 不行

# In[107]:


a = np.arange(9)
print(a.reshape(3, 3, order = "C")) # order = "C" 的 C 是 C語言的縮寫，他是 by column
print(a.reshape(3, 3, order = "F")) # order = "F" 的 F 是 Fortran 語言的縮寫，他是 by row


# #### reshape 是 mutable, resize 是 immutable

# In[108]:


a = np.arange(6)
b = np.reshape(a, (2, 3))
b[0,0] = 99 # 改 b
print(b)
print(a) # a 跟著變


# In[109]:


a = np.arange(6)
b = np.resize(a, (2, 3))
b[0,0] = 99 # 改 b
print(b)
print(a) # a 不會跟著變


# * 如果希望 reshape 也是 immutable，那要善用 copy

# In[117]:


a = np.arange(6)
b = np.reshape(a, (2, 3)).copy()
b[0,0] = 99 # 改 b
print(b)
print(a) # a 不會跟著變


# ### 拉直： flatten (immutable) / ravel (mutable)

# #### 語法相同，但 flatten 是 immutable, ravle 是 mutable

# In[118]:


a = np.arange(9).reshape(3,3)
a


# In[120]:


b = a.flatten()
print(b)
print(a)


# In[121]:


b[0] = 99
print(b)
print(a)


# * 可以看到，flatten 完，要 assign 他給新變數。而且，後續修改b，也不會影響到 a
# * 但如果是 ravel，那改b，會影想到 a

# In[122]:


a = np.arange(9).reshape(3,3)
b = np.ravel(a)

print(b)
print(a)


# In[123]:


b[0] = 99
print(b)
print(a)


# * 最後補充一下，你要用之前學過的 reshape 也可以，只是用 flatten() 比較簡便而已:

# In[157]:


a = np.arange(9).reshape(3,3)
a


# In[158]:


a.reshape(-1) # reshape成一軸而已，維度 np 幫我找就好


# In[159]:


np.reshape(a, -1)


# ### 增加軸數

# ### transpose

# #### 對 2d array 做 transpose

# #### > 2d array 的 transpose

# ## 合併 arrays

# ### np.append()

# * 這邊要注意，array 沒有 `.append()` 這個 method，只有 `np.append()` 這個 function

# #### 1d array

# In[164]:


a = np.arange(4)
b = np.arange(4, 7)

print(a)
print(b)
print(np.append(a, b))


# #### 2d array

# In[165]:


b = np.arange(12).reshape((3,4))
b


# * 如果想做到像 R 那樣的 rbind，那要記得，加進去的 array 軸數要一樣
# * 例如，現在的 b 是 2軸 (因為 shape 是 (3,4)，有兩個 element)，所以，我也要加兩軸的資料進去

# In[168]:


# new = [12, 13, 14, 15] # 這是錯的，因為軸數 = 1，不是原本的軸數 = 2
new = [[12, 13, 14, 15]] # 這是對的，因為軸數 = 2，和要 combind 的軸數一樣。也可簡單想成，這才是 row vector 的 shape
np.append(b, new, axis = 0) # 一定要加 axis = 0, 因為你要沿著 axis = 0 的方向加資料。沒寫的話，他默認 axis = None，會展成一軸


# * 如果想做 R 的 cbind，那方法一樣：

# In[169]:


new = [[12], [13], [14]]
np.append(b, new, axis = 1)


# * 那當然也可兩個 2d-array 做 append

# In[170]:


first = np.arange(12).reshape((3,4))
second = np.arange(12, 24).reshape((3,4))


# In[171]:


# cbind
np.append(first, second, axis = 1)


# In[172]:


# rbind
np.append(first, second, axis = 0)


# * 這邊要講，如何把多個 arrays 組合起來

# #### np.concaterate()

# In[53]:


a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
# a + b # wrong
# np.array([a, b]) # wrong
np.concatenate((a, b))


# In[54]:


x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
np.concatenate((x, y), axis=0) # 沿第0軸，最外層，concate


# #### np.vstack()

# In[88]:


a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])

np.vstack((a1, a2))


# #### np.hstack()

# In[89]:


a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])

np.hstack((a1, a2))


# ## Array 馬殺雞

# ### Splitting

# * 這邊要講，如何把一個 array，拆成多個：

# In[90]:


x = np.arange(1, 25).reshape(2, 12)
x


# #### np.hsplit()

# In[92]:


np.hsplit(x, 3) # 水平均分成 3 份


# In[94]:


np.hsplit(x, (3, 4)) # 我想在 column 3 切一刀， column 4 切一刀


# #### np.vsplit()

# In[97]:


np.vsplit(x, 2) # 垂直均分兩份


# ### flip (reverse)

# #### 1d array

# In[151]:


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
reversed_arr = np.flip(arr)
reversed_arr


# #### 2d array

# In[152]:


arr_2d = np.array([[1, 2, 3, 4], 
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]])
arr_2d


# In[153]:


reversed_arr = np.flip(arr_2d)
print(reversed_arr)


# In[158]:


reversed_arr_rows = np.flip(arr_2d, axis=0) # 只對第0軸reverse，所以原本是 [A, B, C], 變 [C, B, A]，其中 A = [1,2,3,4]
print(reversed_arr_rows)


# In[157]:


reversed_arr_cols = np.flip(arr_2d, axis=1)
print(reversed_arr_cols)


# ### newaxis 與 np.expand_dims()

# * 這邊要介紹，如何把 1d array，轉成 row vecctor / column vector

# In[66]:


a = np.array([1, 2, 3, 4, 5, 6])
a.shape


# In[74]:


a_row_vector = a[np.newaxis, :]
print(a_row_vector) # 變成 row vector
print(a_row_vector.shape)


# * np.newaxis 就是宣告多一個軸，而且是放在第一個位子
# * 如果放在第二個位子，變成 column vector

# In[75]:


a_col_vector = a[:, np.newaxis]
print(a_col_vector)
print(a_col_vector.shape)


# * 也可以用 `np.expand_dims()` 來處理

# In[76]:


a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)
b = np.expand_dims(a, axis=0)
print(b.shape)
c = np.expand_dims(a, axis=1)
print(c.shape)


# ## Arithmetic Operation

# ![](figures/np_sub_mult_divide.png)

# In[98]:


data = np.array([1, 2])
ones = np.ones(2, dtype=int)
print(data - ones)
print(data * data)
print(data / data)


# ## Aggregate Functions

# ### `.sum()`, `.max()`, `.min()`, `.mean()`, `np.median()`, `.std()`

# #### 1d array

# In[101]:


a = np.array([1, 2, 3, 4])
a.sum()


# #### 2d array

# ![](figures/np_matrix_aggregation.png)

# ![](figures/np_matrix_aggregation_row.png)

# In[123]:


b = np.array([[1, 2], 
              [3, 4],
              [5, 6]])
b


# * 只寫 `.sum()`，就是全加

# In[124]:


b.sum()


# * 有指定 axis，就是沿著那個 axis 做相加

# In[103]:


b.sum(axis = 0) #沿著第0軸相加，所以是 [1, 1] + [2, 2]


# In[104]:


b.sum(axis = 1) # 沿著第1軸相加，所以是 1 + 1; 2+2


# In[125]:


print(b.max())
print(b.max(axis = 0))
print(b.max(axis = 1))


# In[117]:


print(b.min())
print(b.min(axis = 0))
print(b.min(axis = 1))


# In[118]:


print(b.mean())
print(b.mean(axis = 0))
print(b.mean(axis = 1))


# In[121]:


# b.median() # wrong，沒有這個 method
print(np.median(b))
print(np.median(b, axis = 0))
print(np.median(b, axis = 1))


# In[122]:


print(b.std())
print(b.std(axis = 0))
print(b.std(axis = 1))


# ### `np.unique()`

# #### 1d array

# In[135]:


a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
a


# In[136]:


unique_values = np.unique(a)
print(unique_values)


# * 如果你想拿到 index (如果有重複的值，只給我第一個出現的 index 就好)，可以這樣做

# In[137]:


unique_values, indices_list = np.unique(a, return_index=True)
print(indices_list)


# * 表示，我如果要取 unique 的值，就從原本的 array 中，取出 [0, 2, ..., 14] 的位子的值就是了

# * 如果我想看每個值重複的狀況，我可以這樣做：

# In[138]:


unique_values, occurrence_count = np.unique(a, return_counts=True)
print(occurrence_count)


# #### 2d array

# In[147]:


a_2d = np.array([[1, 2, 2, 4], 
                 [5, 6, 6, 8], 
                 [9, 10, 10, 12], 
                 [1, 2, 2, 4]])
a_2d


# * 只用 `np.unique()`，就是全部一起看：

# In[148]:


unique_values = np.unique(a_2d)
print(unique_values)


# * 加入 axis，就可以看沿著那軸的 unique

# In[149]:


unique_rows = np.unique(a_2d, axis=0)
print(unique_rows)


# In[150]:


unique_cols = np.unique(a_2d, axis=1)
print(unique_cols)


# ## Working with mathematical formulas

# ![](figures/np_MSE_formula.png)

# ![](figures/np_MSE_implementation.png)

# ![](figures/np_mse_viz1.png)

# ![](figures/np_mse_viz2.png)

# ## 儲存 與 讀取 numpy array

# ### np.save() & np.load()

# In[159]:


a = np.array([1, 2, 3, 4, 5, 6])
np.save('data/filename', a)
b = np.load('data/filename.npy')
print(b)


# ### np.savetxt() & np.loadtxt()

# In[160]:


csv_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
np.savetxt('data/new_file.csv', csv_arr)
np.loadtxt('data/new_file.csv')

