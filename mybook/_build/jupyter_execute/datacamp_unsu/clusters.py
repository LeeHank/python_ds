#!/usr/bin/env python
# coding: utf-8

# # Clustering

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# ## K-Means

# ### toy-example (iris)

# * 出動 iris 資料集吧

# In[31]:


iris = load_iris()
df_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
df_data


# In[36]:


from sklearn.cluster import KMeans


X = df_data.drop("Species", axis = 1)
kmeansModel = KMeans(n_clusters=3, # 分 3 群
                     n_init = 10, # 起始值有10組，每組分完後，比 inertia，選最小的那組給你
                     random_state=46, # 產出 10 組起始值時，用的 seed
                    tol = 0.0001) # 前後兩次 iteration， centers 間的 歐式距離，小於 0.0001 就算收斂
kmeansModel.fit(X)
clusters_pred = kmeansModel.predict(X)


# * 分群結果

# In[37]:


clusters_pred


# * 各群的中心

# In[38]:


kmeansModel.cluster_centers_


# * inertia (within-group sum of variance)

# In[39]:


kmeansModel.inertia_


# * 選 k

# In[41]:


kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_list]


# In[43]:


plt.plot(range(1,10), inertias);


# * 可以看到，elbow 在 2 or 3，所以可以選 2 or 3 當分群的群數

# * 詳細說明文件，看 `KMeans?`

# In[32]:


get_ipython().run_line_magic('pinfo', 'KMeans')


# ### stock movement

# * 這個應用蠻好玩的，我們可以對股票的 "走勢" 做分群。
# * 資料如以下： 

# In[19]:


movements = pd.read_csv("data/company-stock-movements.csv", index_col = 0)
movements.head()


# * 可以看到，每一列是一家公司，每一行是時間點，值是股價(大概做過一些調整了，所以有正有負，我們可以不用管他，就當是股價即可)
# * 如果我今天要做的分群，是對絕對的數值做分群，那我就直接用這張表分群就好. 
# * 但如果我今天是想對 "走勢" 做分群，那我會希望對 "列" 做標準化。
#   * 舉例來說，台積電的股價變化是 600, 580, 600, 620, 640，啟基是 60, 58, 60, 62, 64。
#   * 那從 "走勢" 來看，台積跟啟基走勢是一樣的，應該被分為一群，但如果直接做 kmeans，就再見了，因為光 600 和 60 的距離就很遠。  
#   * 另外，台積股價的變化是 -20, 20, 20, 20; 啟基是 -2, 2, 2, 2，這個變動差距也不同，但如果改成看變化百分比(把股價放分母，變化當分子)，那兩邊就又差不多了. 
#   * 所以，我如果先對列做標準化，那兩個公司的數值就都變成 [-0.39, -1.37, -0.39, 0.58, 1.56]，一模模一樣樣，euclidean distance 變成 0，分群時一定放在一塊兒
# * 所以，這一個例子，我們就要對 "列" 做標準化，那就要用到 `Normalizer` 這個 preprocessor:

# In[29]:


from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)


# In[16]:


# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 
                   'companies': movements.index})

# Display df sorted by cluster label
df.sort_values(["labels"])


# In[ ]:




