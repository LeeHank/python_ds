#!/usr/bin/env python
# coding: utf-8

# # Visualization with Seaborn

# * Matplotlib的缺點包括：  
#   * API相對低階，經常需要非常多的code才能畫出一張圖
#   * Matplotlib 比 Pandas 早10年出現，所以他畫圖的 input 是 numpy 的 array，而不是 pandas 的 DataFrame。這樣每次畫圖時，都要從 DataFrame 中取出 series 來畫圖，很麻煩
# * 針對這些問題的解法，就是 `seaborn`，他是建立在 Matplotlib 上面的 API，他提供很多高階函數，而且也可以和 Pandas DataFrame 結合得很好

# ## Seaborn Versus Matplotlib

# In[1]:


import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd


# * 來畫一張 random walk 的線圖：

# In[2]:


# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


# * 這張圖雖然畫出我要的資訊，但蠻醜的，而且有滿滿過時感. 
# * 現在改成用 seaborn

# In[3]:


import seaborn as sns
sns.set() # 設定成 seaborn的style


# Now let's rerun the same two lines as before:

# In[4]:


# 寫法和剛剛一樣
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


# Ah, much better!

# ## 探索 Seaborn 的圖表

# * Seaborn 的主要概念，是提供高階的指令，來建立各式各樣的圖表。尤其在統計學資料探索上非常有用。

# In[5]:


# 生資料，並轉成 DataFrame
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

# 畫圖
for col in 'xy':
    plt.hist(data[col], alpha=0.5)


# * 除了畫 histogram，我們也可以畫 density。作法是，使用 `sns.kdeplot`：

# In[6]:


for col in 'xy':
    sns.kdeplot(data[col], shade=True)


# Histograms 和 density 可以 combine 在一起，用 `distplot`

# In[7]:


sns.distplot(data['x'])
sns.distplot(data['y']);


# * 我們可以同時看 joint distribution 和 marginal distribution，用 `sns.jointplot` 

# In[8]:


with sns.axes_style('white'): # 把背景設為白色
    sns.jointplot("x", "y", data, kind='kde');


# There are other parameters that can be passed to ``jointplot``—for example, we can use a hexagonally based histogram instead:

# In[9]:


with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')


# ### Pair plots

# In[10]:


iris = sns.load_dataset("iris")
iris.head()


# 用 `sns.pairplot` 來繪製 pairplot

# In[11]:


sns.pairplot(iris);


# ### Faceted histograms

# * Seaborn's `FacetGrid` 效果就和 ggplot 的 facet_grid 一樣，只是它是用 oop 寫法，所以要適應一下
# * 我們讀取另一個資料集來示範:

# In[12]:


tips = sns.load_dataset('tips')
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']
tips.head()


# * `tip` 就是小費的意思，`tip_pct` 是小費給幾%。  
# * 我們想 by sex x time 來畫 `tip_pct` 的 histogram，看看午餐晚餐時，男生女生給的小費有沒有差

# In[13]:


grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15));


# * 看起來，好像都差不多，只看得到數量的差別而已(男生比女生消費次數多、晚餐比午餐消費次數多)

# ### Factor plots
# 
# Factor plots can be useful for this kind of visualization as well. This allows you to view the distribution of a parameter within bins defined by any other parameter:

# In[14]:


with sns.axes_style(style='ticks'):
        g = sns.catplot(
            x = "day", 
            y = "total_bill", 
            hue = "sex", 
            data=tips, 
            kind="box")
        g.set_axis_labels("Day", "Total Bill");


# ### Joint distributions
# 
# Similar to the pairplot we saw earlier, we can use ``sns.jointplot`` to show the joint distribution between different datasets, along with the associated marginal distributions:

# In[15]:


with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex')


# The joint plot can even do some automatic kernel density estimation and regression:

# In[16]:


sns.jointplot("total_bill", "tip", data=tips, kind='reg');


# ### Bar plots
# 
# Time series can be plotted using ``sns.factorplot``. In the following example, we'll use the Planets data that we first saw in [Aggregation and Grouping](03.08-Aggregation-and-Grouping.ipynb):

# In[17]:


planets = sns.load_dataset('planets')
planets.head()


# In[18]:


with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=5)


# We can learn more by looking at the *method* of discovery of each of these planets:

# In[19]:


with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')


# For more information on plotting with Seaborn, see the [Seaborn documentation](http://seaborn.pydata.org/), a [tutorial](http://seaborn.pydata.org/
# tutorial.htm), and the [Seaborn gallery](http://seaborn.pydata.org/examples/index.html).

# ## Example: Exploring Marathon Finishing Times
# 
# Here we'll look at using Seaborn to help visualize and understand finishing results from a marathon.
# I've scraped the data from sources on the Web, aggregated it and removed any identifying information, and put it on GitHub where it can be downloaded
# (if you are interested in using Python for web scraping, I would recommend [*Web Scraping with Python*](http://shop.oreilly.com/product/0636920034391.do) by Ryan Mitchell).
# We will start by downloading the data from
# the Web, and loading it into Pandas:

# In[20]:


# !curl -O https://raw.githubusercontent.com/jakevdp/marathon-data/master/marathon-data.csv


# In[21]:


data = pd.read_csv('marathon-data.csv')
data.head()


# By default, Pandas loaded the time columns as Python strings (type ``object``); we can see this by looking at the ``dtypes`` attribute of the DataFrame:

# In[24]:


data.dtypes


# Let's fix this by providing a converter for the times:

# In[25]:


import datetime

def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return datetime.timedelta(hours=h, minutes=m, seconds=s)

data = pd.read_csv('marathon-data.csv',
                   converters={'split':convert_time, 'final':convert_time})
data.head()


# In[26]:


data.dtypes


# That looks much better. For the purpose of our Seaborn plotting utilities, let's next add columns that give the times in seconds:

# In[27]:


data['split_sec'] = data['split'].astype(int) / 1E9
data['final_sec'] = data['final'].astype(int) / 1E9
data.head()


# To get an idea of what the data looks like, we can plot a ``jointplot`` over the data:

# In[28]:


with sns.axes_style('white'):
    g = sns.jointplot("split_sec", "final_sec", data, kind='hex')
    g.ax_joint.plot(np.linspace(4000, 16000),
                    np.linspace(8000, 32000), ':k')


# The dotted line shows where someone's time would lie if they ran the marathon at a perfectly steady pace. The fact that the distribution lies above this indicates (as you might expect) that most people slow down over the course of the marathon.
# If you have run competitively, you'll know that those who do the opposite—run faster during the second half of the race—are said to have "negative-split" the race.
# 
# Let's create another column in the data, the split fraction, which measures the degree to which each runner negative-splits or positive-splits the race:

# In[29]:


data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']
data.head()


# Where this split difference is less than zero, the person negative-split the race by that fraction.
# Let's do a distribution plot of this split fraction:

# In[30]:


sns.distplot(data['split_frac'], kde=False);
plt.axvline(0, color="k", linestyle="--");


# In[31]:


sum(data.split_frac < 0)


# Out of nearly 40,000 participants, there were only 250 people who negative-split their marathon.
# 
# Let's see whether there is any correlation between this split fraction and other variables. We'll do this using a ``pairgrid``, which draws plots of all these correlations:

# In[32]:


g = sns.PairGrid(data, vars=['age', 'split_sec', 'final_sec', 'split_frac'],
                 hue='gender', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();


# It looks like the split fraction does not correlate particularly with age, but does correlate with the final time: faster runners tend to have closer to even splits on their marathon time.
# (We see here that Seaborn is no panacea for Matplotlib's ills when it comes to plot styles: in particular, the x-axis labels overlap. Because the output is a simple Matplotlib plot, however, the methods in [Customizing Ticks](04.10-Customizing-Ticks.ipynb) can be used to adjust such things if desired.)
# 
# The difference between men and women here is interesting. Let's look at the histogram of split fractions for these two groups:

# In[33]:


sns.kdeplot(data.split_frac[data.gender=='M'], label='men', shade=True)
sns.kdeplot(data.split_frac[data.gender=='W'], label='women', shade=True)
plt.xlabel('split_frac');


# The interesting thing here is that there are many more men than women who are running close to an even split!
# This almost looks like some kind of bimodal distribution among the men and women. Let's see if we can suss-out what's going on by looking at the distributions as a function of age.
# 
# A nice way to compare distributions is to use a *violin plot*

# In[34]:


sns.violinplot("gender", "split_frac", data=data,
               palette=["lightblue", "lightpink"]);


# This is yet another way to compare the distributions between men and women.
# 
# Let's look a little deeper, and compare these violin plots as a function of age. We'll start by creating a new column in the array that specifies the decade of age that each person is in:

# In[35]:


data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))
data.head()


# In[36]:


men = (data.gender == 'M')
women = (data.gender == 'W')

with sns.axes_style(style=None):
    sns.violinplot("age_dec", "split_frac", hue="gender", data=data,
                   split=True, inner="quartile",
                   palette=["lightblue", "lightpink"]);


# Looking at this, we can see where the distributions of men and women differ: the split distributions of men in their 20s to 50s show a pronounced over-density toward lower splits when compared to women of the same age (or of any age, for that matter).
# 
# Also surprisingly, the 80-year-old women seem to outperform *everyone* in terms of their split time. This is probably due to the fact that we're estimating the distribution from small numbers, as there are only a handful of runners in that range:

# In[38]:


(data.age > 80).sum()


# Back to the men with negative splits: who are these runners? Does this split fraction correlate with finishing quickly? We can plot this very easily. We'll use ``regplot``, which will automatically fit a linear regression to the data:

# In[37]:


g = sns.lmplot('final_sec', 'split_frac', col='gender', data=data,
               markers=".", scatter_kws=dict(color='c'))
g.map(plt.axhline, y=0.1, color="k", ls=":");


# Apparently the people with fast splits are the elite runners who are finishing within ~15,000 seconds, or about 4 hours. People slower than that are much less likely to have a fast second split.

# <!--NAVIGATION-->
# < [Geographic Data with Basemap](04.13-Geographic-Data-With-Basemap.ipynb) | [Contents](Index.ipynb) | [Further Resources](04.15-Further-Resources.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.14-Visualization-With-Seaborn.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
