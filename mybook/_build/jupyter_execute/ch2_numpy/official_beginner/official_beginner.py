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
#   * 我收到 100 個人的身高和體重，我可以儲存成 2d-array: (e.g. [[172,...,158],[65,...,54]]) -> $R^{100人 \times 2變數}$
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

# In[2]:


# 1d-array
a = np.array([1,2,3])
a


# * 因為所有的 element 都是數字，所以我可以做數學計算：

# In[3]:


a.mean()


# ### 2d-array

# * 對於1張灰階影像資料，例如是這樣的一張矩陣型資料: $\left[\begin{array}{cc} 0 & 1\\1 & 0 \\1 & 1\end{array}\right]$，可以用數學寫成： $\boldsymbol{X} \in R^{3 \times 2}$ 
# * 在 python 中，會用這樣的 array 來儲存他：  

# In[4]:


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

# In[5]:


# 第一步，先寫出第一層，有3列： 
# a = [[], [], []] 

# 第二步，再把第二層的內容補進去，各2個element： 
a = [[0, 1], [1, 0], [1, 1]]

# 第三步，轉成 np.array
a = np.array(a)


# * 接著，來定義一些名詞： $R^{3 \times 2}$，R的上面有`2`個數字相乘，我們稱它為`2`階張量，儲存的資料類型是 `2`d array。也就是說，這個張量的`維度是2`。然後 R 上面的長相是 $3 \times 2$，所以我們說他的 shape 是 `(3,2)`  
# * 我們來看一下這個 numpy array 的 attribute，就可以驗證上面講的內容：

# In[6]:


a.ndim


# * ndim 是 2，就表示 ndarray 是 2d array(n=2, 有兩層，R上面有2個數字相乘)  

# In[7]:


a.shape


# * shape 是 (3,2)，表示他是 $R^{3 \times 2}$ 的張量  

# ### 3d-array

# In[8]:


# 2d-array
b = np.array([[1,2],[3,4]])
b


# In[9]:


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

# In[10]:


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

# In[11]:


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

# In[12]:


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

# In[13]:


print(f"the dim of a is {a.ndim}")
print(f"the shape of a is {a.shape}")
print(a.size)
len(a)


# ## 建立特殊 array 

# * 我們可以造出各種特殊的 array
#   * 造出 constant array (e.g. `np.zeros()`, `np.ones()`, `np.full()`  
#   * 做出單位矩陣 (e.g. `np.eye()`)
#   * 做出等距的向量 (e.g. `np.arange()`, `np.linspace()`)
#   * 做出隨機向量 (e.g. `np.random.*()`)

# ### np.zeros()

# In[14]:


np.zeros(4)


# In[15]:


np.zeros((3,2))


# ### np.ones()

# In[16]:


np.ones(4)


# In[17]:


np.ones((3,2))


# ### np.full()

# In[18]:


np.full(5, 4) # 做一個 shape = 5 的 array，element 都是 4


# In[19]:


np.full((3,2), 4) # 做一個 shape = (3,2) 的 array，element 都是 4


# ### np.arange()

# * create an array of evenly spaced values (step value)

# In[20]:


np.arange(10)


# In[21]:


np.arange(10,25,3) # 第三個 argument, 是步數, 也就是間隔 3


# ### np.linspace()

# * create an array of evenly spaced values (number of samples)

# In[22]:


np.linspace(10,25,3) # 第三個 argument, 是總數


# In[23]:


np.random.random((2,2))


# In[24]:


get_ipython().run_line_magic('pinfo', 'np.random.random')


# ### np.random.*()

# * np.random.random() 可產出 0 ~ 1 的亂數. 
# * 可用 `np.random.random?` 來查詢用法. 
# * 可用 `np.random.<tab>` 找到還有一堆隨機分配可以用，例如 `np.random.normal()`

# In[25]:


np.random.random(5)


# In[26]:


np.random.random((3,2))


# ## Array 馬殺雞

# ### Sorting

# In[27]:


a = np.array([3,6,2])
a.sort() # a 已變，不用再 assign 回 a
a


# In[28]:


c = np.array([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [3, 2, 1],
        [7, 8, 9]
    ]
])

c


# ### Combining arrays

# * 這邊要講，如何把多個 arrays 組合起來

# #### np.concaterate()

# In[29]:


a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
# a + b # wrong
# np.array([a, b]) # wrong
np.concatenate((a, b))


# In[30]:


x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
np.concatenate((x, y), axis=0) # 沿第0軸，最外層，concate


# #### np.vstack()

# In[31]:


a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])

np.vstack((a1, a2))


# #### np.hstack()

# In[32]:


a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])

np.hstack((a1, a2))


# ### Splitting

# * 這邊要講，如何把一個 array，拆成多個：

# In[33]:


x = np.arange(1, 25).reshape(2, 12)
x


# #### np.hsplit()

# In[34]:


np.hsplit(x, 3) # 水平均分成 3 份


# In[35]:


np.hsplit(x, (3, 4)) # 我想在 column 3 切一刀， column 4 切一刀


# #### np.vsplit()

# In[36]:


np.vsplit(x, 2) # 垂直均分兩份


# ### Reshape

# ![](figures/np_reshape.png)

# In[37]:


a = np.arange(6)
print(a)


# In[38]:


a.reshape(3, 2) # 轉成 shape = (3,2), 他會 byrow 填入


# In[39]:


a.reshape(6) # 轉成 shape = (6,)


# ### flatten / ravel

# * 用 flatten 時，你如果改新的 array，不會影響到舊的 (immutable)

# In[40]:


x = np.array([[1 , 2, 3, 4], 
              [5, 6, 7, 8], 
              [9, 10, 11, 12]])


# In[41]:


print(x.flatten())
print(x) # x 不變，所以 flatten 完應該要存成新變數


# In[42]:


a1 = x.flatten()
a1[0] = 99 # 改了新的
print(x)  # 舊的不變
print(a1)  # New array


# * 但如果你用 `.ravel()`，那舊的會跟著變

# In[43]:


a2 = x.ravel()
a2[0] = 98
print(x)  # Original array
print(a2)  # New array


# ### flip (reverse)

# #### 1d array

# In[44]:


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
reversed_arr = np.flip(arr)
reversed_arr


# #### 2d array

# In[45]:


arr_2d = np.array([[1, 2, 3, 4], 
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]])
arr_2d


# In[46]:


reversed_arr = np.flip(arr_2d)
print(reversed_arr)


# In[47]:


reversed_arr_rows = np.flip(arr_2d, axis=0) # 只對第0軸reverse，所以原本是 [A, B, C], 變 [C, B, A]，其中 A = [1,2,3,4]
print(reversed_arr_rows)


# In[48]:


reversed_arr_cols = np.flip(arr_2d, axis=1)
print(reversed_arr_cols)


# ### newaxis 與 np.expand_dims()

# * 這邊要介紹，如何把 1d array，轉成 row vecctor / column vector

# In[49]:


a = np.array([1, 2, 3, 4, 5, 6])
a.shape


# In[50]:


a_row_vector = a[np.newaxis, :]
print(a_row_vector) # 變成 row vector
print(a_row_vector.shape)


# * np.newaxis 就是宣告多一個軸，而且是放在第一個位子
# * 如果放在第二個位子，變成 column vector

# In[51]:


a_col_vector = a[:, np.newaxis]
print(a_col_vector)
print(a_col_vector.shape)


# * 也可以用 `np.expand_dims()` 來處理

# In[52]:


a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)
b = np.expand_dims(a, axis=0)
print(b.shape)
c = np.expand_dims(a, axis=1)
print(c.shape)


# ## subsetting, slicing, indexing

# ![](figures/np_indexing.png)

# ### subsetting

# In[53]:


data = np.array([1, 2, 3])
data[0]


# ### slicing

# In[54]:


data[0:2]


# In[55]:


data[1:]


# In[56]:


data[-2:]


# ### booling indexing

# In[57]:


a = np.array([[1 , 2, 3, 4], 
              [5, 6, 7, 8], 
              [9, 10, 11, 12]])
a


# In[58]:


a < 5


# In[59]:


a[a < 5]


# In[60]:


a[(a > 2) & (a < 11)]


# ## Broadcasting

# ![](figures/np_multiply_broadcasting.png)

# In[61]:


data = np.array([1.0, 2.0])
data * 1.6


# ![](figures/np_matrix_broadcasting.png)

# In[62]:


data = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([[1, 1]])
data + ones_row


# ## Arithmetic Operation

# ![](figures/np_sub_mult_divide.png)

# In[63]:


data = np.array([1, 2])
ones = np.ones(2, dtype=int)
print(data - ones)
print(data * data)
print(data / data)


# ## Aggregate Functions

# ### `.sum()`, `.max()`, `.min()`, `.mean()`, `np.median()`, `.std()`

# #### 1d array

# In[64]:


a = np.array([1, 2, 3, 4])
a.sum()


# #### 2d array

# ![](figures/np_matrix_aggregation.png)

# ![](figures/np_matrix_aggregation_row.png)

# In[65]:


b = np.array([[1, 2], 
              [3, 4],
              [5, 6]])
b


# * 只寫 `.sum()`，就是全加

# In[66]:


b.sum()


# * 有指定 axis，就是沿著那個 axis 做相加

# In[67]:


b.sum(axis = 0) #沿著第0軸相加，所以是 [1, 1] + [2, 2]


# In[68]:


b.sum(axis = 1) # 沿著第1軸相加，所以是 1 + 1; 2+2


# In[69]:


print(b.max())
print(b.max(axis = 0))
print(b.max(axis = 1))


# In[70]:


print(b.min())
print(b.min(axis = 0))
print(b.min(axis = 1))


# In[71]:


print(b.mean())
print(b.mean(axis = 0))
print(b.mean(axis = 1))


# In[72]:


# b.median() # wrong，沒有這個 method
print(np.median(b))
print(np.median(b, axis = 0))
print(np.median(b, axis = 1))


# In[73]:


print(b.std())
print(b.std(axis = 0))
print(b.std(axis = 1))


# ### `np.unique()`

# #### 1d array

# In[74]:


a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
a


# In[75]:


unique_values = np.unique(a)
print(unique_values)


# * 如果你想拿到 index (如果有重複的值，只給我第一個出現的 index 就好)，可以這樣做

# In[76]:


unique_values, indices_list = np.unique(a, return_index=True)
print(indices_list)


# * 表示，我如果要取 unique 的值，就從原本的 array 中，取出 [0, 2, ..., 14] 的位子的值就是了

# * 如果我想看每個值重複的狀況，我可以這樣做：

# In[77]:


unique_values, occurrence_count = np.unique(a, return_counts=True)
print(occurrence_count)


# #### 2d array

# In[78]:


a_2d = np.array([[1, 2, 2, 4], 
                 [5, 6, 6, 8], 
                 [9, 10, 10, 12], 
                 [1, 2, 2, 4]])
a_2d


# * 只用 `np.unique()`，就是全部一起看：

# In[79]:


unique_values = np.unique(a_2d)
print(unique_values)


# * 加入 axis，就可以看沿著那軸的 unique

# In[80]:


unique_rows = np.unique(a_2d, axis=0)
print(unique_rows)


# In[81]:


unique_cols = np.unique(a_2d, axis=1)
print(unique_cols)


# ## Working with mathematical formulas

# ![](figures/np_MSE_formula.png)

# ![](figures/np_MSE_implementation.png)

# ![](figures/np_mse_viz1.png)

# ![](figures/np_mse_viz2.png)

# ## 儲存 與 讀取 numpy array

# ### np.save() & np.load()

# In[82]:


a = np.array([1, 2, 3, 4, 5, 6])
np.save('data/filename', a)
b = np.load('data/filename.npy')
print(b)


# ### np.savetxt() & np.loadtxt()

# In[83]:


csv_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
np.savetxt('data/new_file.csv', csv_arr)
np.loadtxt('data/new_file.csv')

