#!/usr/bin/env python
# coding: utf-8

# # Introduction to Seaborn

# * seaborn 是 build 在 matplotlib 上，和 pandas 合作的很好

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# ## Scatter plot

# ### Data

# #### student_data

# In[2]:


student_data = pd.read_csv("data/student-alcohol-consumption.csv", index_col=0)
student_data


# #### mpg

# In[3]:


mpg = pd.read_csv("data/mpg.csv")
mpg


# ### 基本 scatter plot

# In[4]:


sns.scatterplot(x="absences", y="G3", 
                data=student_data);


# * G3 是 第三次段考的意思。可以看到，缺席率越高，看起來成績越低

# ### hue (i.e. color)

# In[5]:


sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location");


# * 可以看到，第三軸放上 location 後，結論是：不論是城市或鄉下的小孩，都是缺席越多，成績越差

# #### 第三個變數的顏色自己指定

# In[6]:


hue_colors = {
    "Urban": "black",
    "Rural": "red"
}
sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location",
                hue_order = ["Rural", "Urban"],
                palette = hue_colors);


# #### 第三個變數的順序自己指定

# In[7]:


sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location",
                hue_order = ["Rural", "Urban"]); # 先 Rural 再 Urban


# ### size

# In[8]:


sns.scatterplot(
    x="horsepower", 
    y="mpg",
    data=mpg,
    size="cylinders"
);


# * 可以看到，汽缸數越多(cylinders)，horsepower越大，而油耗越差(mpg)

# In[9]:


sns.scatterplot(
    x="horsepower", 
    y="mpg",
    data=mpg,
    size="cylinders",
    hue = "cylinders"
);


# * 加上顏色，看得更清楚
# * 也因為 cylinders 被他認為是 float 變數，所以當第三軸的顏色時，他是給你 gradient 顏色，比較好觀察

# ### style (點的style)

# In[10]:


sns.scatterplot(
    x="acceleration", 
    y="mpg",
    data=mpg,
    style ="origin",
    hue = "origin"
);


# * 可以看到，usa的車子最多，而且比起 japan 和 europe 的特色，是他有一部分都聚在左下角：表示加速快 & 油耗差

# In[11]:


sns.scatterplot(
    x = "absences",
    y = "G3",
    data = student_data,
    style = "traveltime",
    hue = "traveltime"
);


# ### alpha

# ### facet_grid 類型

# * ggplot 的 facet_grid/facet_wrap，在 sns 中，是用 `relplot()` 來實現
# * relplot 是 relational plot 的縮寫，它包含 scatter plot 和 line plot. 
# * 我們使用 relplot 的時機是，你想做出 ggplot 那種 facet_wrap 的效果

# #### by column 畫圖

# In[12]:


sns.relplot(x="absences", 
            y="G3",
            data=student_data, 
            kind = "scatter",
            col = "location");


# #### by column 指定順序

# In[13]:


sns.relplot(x="absences", 
            y="G3",
            data=student_data, 
            kind = "scatter",
            col = "location",
           col_order = ["Rural", "Urban"]);


# #### 指定 column 行數

# * 可以定義 by col 畫圖時，最多幾個後要換行

# In[14]:


sns.relplot(x="absences", 
            y="G3",
            data=student_data, 
            kind = "scatter",
            col = "study_time",
            col_wrap = 2);


# #### by row 畫圖

# * 同樣的做法，可以改成 by row

# In[15]:


sns.relplot(x="absences", 
            y="G3",
            data=student_data, 
            kind = "scatter",
            row = "location");


# #### by column & row (R 的 facet_grid)

# * 如果要做到 facet_grid (兩個變數交叉)，那就又 col 又 row

# In[16]:


sns.relplot(x="absences", 
            y="G3",
            data=student_data, 
            kind = "scatter",
            col = "study_time",
           row = "location");


# * 當然，剛剛 row 和 column 用過的細節設定都還是可以下：

# In[17]:


sns.relplot(x="G1", y="G3", # 第一學期 和 第三學期 的成績
            data=student_data,
            kind="scatter", 
            col="schoolsup", # 有沒有獲得學校補助 school support
            col_order=["yes", "no"],
            row = "famsup", # 有沒有獲得家庭補助 family support
            row_order = ["yes", "no"])


# ## Line plot

# ### 基本 line plot

# ### multiple line plot

# ### line plot with CI

# ## count plots (bar chart)

# ### 基本 countplot

# In[19]:


sns.countplot(x = "school", 
              data = student_data);


# ### 兩維的 countplot

# In[20]:


palette_colors = {"Rural": "green", "Urban": "blue"}

sns.countplot(x = "school", 
              data = student_data, 
              hue = "location", 
              palette = palette_colors);


# In[22]:


countries = pd.read_csv("data/countries-of-the-world.csv")
countries


# In[9]:


countries.info()


# In[23]:


sns.scatterplot(x = "GDP ($ per capita)", y = "Literacy (%)", data = countries);


# ## count plot
