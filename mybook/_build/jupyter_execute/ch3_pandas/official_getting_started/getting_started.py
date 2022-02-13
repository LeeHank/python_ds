#!/usr/bin/env python
# coding: utf-8

# # Getting Started (official)

# * 這份文件，摘錄自官網的 [Getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt


# ## Pandas 處理哪種 data？

# * 處理 tabular data ，如下：

# ![](figures/schemas/01_table_dataframe.svg)

# ### DataFrame

# * 建立 DataFrame 的方式，由 dictionary 來處理：

# In[6]:


df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)

print(df)
print(type(df))


# * 這其實就和 excel 一樣:

# ![](figures/schemas/01_table_spreadsheet.png)

# ### Series

# * 每一個 column，都是一個 `Series`

# ![](figures/schemas/01_table_series.svg)

# * 要注意，series還是帶有 row index。例如：

# In[7]:


df["Age"]


# * 如果，你想自己建立一個 series，可以這樣做：

# In[8]:


ages = pd.Series([22, 35, 58], name="Age")
ages


# ### 一些基礎 methods

# * 對 Series，我們可以進行 numpy 的那些常見 function，例如：

# In[9]:


df["Age"].max()


# * 對 DataFrame，我們可以看一下連續型欄位的 basic statistics

# In[10]:


df.describe()


# ## 如何 讀寫 tabular data？

# ![](figures/schemas/02_io_readwrite.svg)

# * 如上圖， `read_*` 就讀進來，用 `to_*` 就寫出去

# ### csv

# In[13]:


titanic = pd.read_csv("data/titanic.csv")
titanic.head()


# In[14]:


titanic.tail()


# ### excel

# * 記得要先安裝 `openpyxl` ，才能順利讀寫  
# * 我們可以把剛剛的 `titanic` DataFrame 先寫成 excel

# In[16]:


titanic.to_excel("data/titanic.xlsx", sheet_name="passengers", index=False)


# * 然後，我們把他讀進來看看

# In[17]:


titanic = pd.read_excel("data/titanic.xlsx", sheet_name="passengers")
titanic.head()


# * 看一下這張 table 的欄位摘要：

# In[18]:


titanic.info()


# * 第一列告訴你他的型別是 `DataFrame`. 
# * 第二列告訴你他的 row index 從 0 ~ 890 (共 891 個 row)
# * 第三列告訴你有 12 個 column. 
# * 第四列開始，摘要每個欄位的資訊. 
#   * Non-Null Count 可以讓你看到大部分的 column都沒有 missing (891 non-null)，但 `Age`, `Cabin`, `Embarked` 有 missing  
#   * Dtype 可以讓你看到每個 column 的 type。`object` 的意思，就是文字型/類別型資料; `int64` 是整數型資料，`float64` 是real number型資料  
# * 倒數第二列，幫你摘要變數的 type，對統計分析來說，就知道數值型資料有 7 個 (float64 + int64)，類別型有 5 個 (object)
# * 最後一列告訴你 memory usage 是 84 kb 左右

# ## 如何 select/filter

# ### select 特定 column

# ![](figures/schemas/03_subset_columns.svg)

# In[20]:


age_sex = titanic[["Age", "Sex"]]
age_sex.head()


# ### filter 特定 row

# ![](figures/schemas/03_subset_rows.svg)

# In[21]:


above_35 = titanic[titanic["Age"] > 35]
above_35.head()


# In[22]:


class_23 = titanic[titanic["Pclass"].isin([2, 3])]
class_23.head()


# In[23]:


age_no_na = titanic[titanic["Age"].notna()]
age_no_na.head()


# ### select + filter

# ![](figures/schemas/03_subset_columns_rows.svg)

# In[24]:


adult_names = titanic.loc[titanic["Age"] > 35, "Name"]
adult_names.head()


# In[25]:


titanic.iloc[9:25, 2:5]


# ## 如何畫圖

# * 這一章，我們拿 air quality 的資料集來舉例

# In[27]:


air_quality = pd.read_csv("data/air_quality_no2.csv", index_col=0, parse_dates=True)
air_quality.head()


# ![](figures/schemas/04_plot_overview.svg)

# * 在 pandas 中，只要做 `DataFrame.plot.*` 就可以畫圖，這個星號包括以下幾種：  
#   * `df.plot()`: 對每個 column 畫 line plot. 
#   * `series.plot()`: 對這個serieis 畫 line plot
#   * `df.plot.scatter(x,y)`: x-y 散布圖
#   * `df.plot.box()`
#   * `df.plot.hist()`  
#   * `df.plot.bar()`. 
#   * `df.plot.line()` 
#   * `df.plot.kde()` 
#   * `df.plot.density()` 
#    
# * 如果想知道到底可以畫哪些圖，可以用 `df.plot.<tab>` 就可以知道有哪些 method 可以用
# * 這邊條列所有可用的 method 如下：

# In[33]:


[method_name for method_name in dir(air_quality.plot) if not method_name.startswith("_")]


# ### 對整個 df 畫圖 (各欄位的line plot)

# * 我如果直接用 `air_quality.plot()`，那預設的作法是：對每一個 column 都去畫 line plot
# * 所以以這個資料集為例，就會畫出3條 time-series plot

# In[29]:


air_quality.plot();


# ### 對某個 series 畫圖 (該series line plot)

# In[31]:


air_quality["station_paris"].plot();


# ### Scatter plot

# In[32]:


air_quality.plot.scatter(x="station_london", y="station_paris", alpha=0.5);


# ### Box plot

# In[35]:


air_quality.plot.box(); # 對每個 column 畫圖


# ### Area plot

# In[37]:


air_quality.plot.area();


# ### subplot

# * 如果我想做成 subplot，可以這樣做：

# In[40]:


air_quality.plot.area(subplots = True);


# ### 更多客製化

# * 如果要做更多客製化，那就要用 matplotlib 的 oop 的寫法

# In[45]:


fig, axs = plt.subplots(figsize=(12, 4));
air_quality.plot.area(ax=axs); # 建立連結，pandas畫完的圖，本來就是 matplotlib 物件，現在告訴他我要把這物件更新到外面的 axs
axs.set_ylabel("NO$_2$ concentration");
# fig.savefig("no2_concentrations.png")


# ### 更多細節

# * 更多畫圖的細節，請參考 user guide 的 [chart Visualization](https://pandas.pydata.org/docs/user_guide/visualization.html#)

# ## 如何新增 column

# ### 某個 column 乘上一個常數

# ![](figures/schemas/05_newcolumn_1.svg)

# * 如上圖，我想新增 column，我可以這樣做：

# In[46]:


air_quality["london_mg_per_cubic"] = air_quality["station_london"] * 1.882
air_quality.head()


# ### 多個 column 間的運算

# ![](figures/schemas/05_newcolumn_2.svg)

# * 那如果是像上圖，我要用兩個欄位來計算出新欄位，我可以這樣做：

# In[47]:


air_quality["ratio_paris_antwerp"] = (
    air_quality["station_paris"] / air_quality["station_antwerp"]
)

air_quality.head()


# ### rename

# * 要對 column name 做 rename 的話，可以這樣做

# In[48]:


air_quality_renamed = air_quality.rename(
    columns={
        "station_antwerp": "BETR801",
        "station_paris": "FR04014",
        "station_london": "London Westminster",
    }
)

air_quality_renamed.head()


# * 我也可以用 function，把 column name 都轉小寫：

# In[49]:


air_quality_renamed = air_quality_renamed.rename(columns=str.lower)
air_quality_renamed.head()


# ## 如何做 summary statistics

# ### aggregating statistics

# ![](figures/schemas/06_aggregate.svg)

# In[50]:


titanic["Age"].mean()


# ![](figures/schemas/06_reduction.svg)

# In[51]:


titanic[["Age", "Fare"]].median()


# In[52]:


titanic[["Age", "Fare"]].describe()


# In[53]:


titanic.agg(
    {
        "Age": ["min", "max", "median", "skew"],
        "Fare": ["min", "max", "median", "mean"],
    }
)


# ### by group

# ![](figures/schemas/06_groupby_select_detail.svg)

# In[58]:


(
    titanic
    .groupby("Sex")
    ["Age"]
    .mean()
)


# ![](figures/schemas/06_groupby_agg_detail.svg)

# In[59]:


titanic.groupby("Sex").mean() # 對所有 numeric column 取 mean


# In[60]:


titanic.groupby(["Sex", "Pclass"])["Fare"].mean()


# ### count number

# * 如果我只是想看某個類別變數 (e.g. `Pclass`) 的次數分配，那可以這樣做：

# In[61]:


titanic["Pclass"].value_counts()


# * 那其實我也可以這樣做：

# In[62]:


titanic.groupby("Pclass")["Pclass"].count()


# * 上面的過程，就如下圖：

# ![](figures/schemas/06_valuecounts.svg)

# ## 如何 reshape

# ### Long to wide (pivot)(R的pivot wider)

# * 我們來看一下 long data 的範例 (就是 stack data 啦)

# In[73]:


air_quality = pd.read_csv(
    "data/air_quality_long.csv", parse_dates=True
)
air_quality = air_quality[air_quality["parameter"]=="no2"]
air_quality.sort_values(["country","city","date.utc"])


# * 從上表可以看到，每一列的 key 是 country + city + date.utc + location，表示該城市在該時間點的該測站，所測到的數值
# * 那 location 就是被我堆疊起來的變數，我想把 location 來成 column，我可以這樣做

# In[79]:


air_quality_wide = air_quality.pivot(
    index = ["city", "country", "date.utc"],
    columns = "location",
    values = "value"
).reset_index() # 如果不 reset_index 的話， city, country, date.utc 會被放在 index
air_quality_wide


# * 看說明文件，可以看到更多例子：

# In[77]:


get_ipython().run_line_magic('pinfo', 'air_quality.pivot')


# ### wide to long (melt)(R的pivot_longer)

# * 回顧剛剛的 wide data：

# In[80]:


air_quality_wide


# * 我現在想倒過來，把 BETR801~London 這幾個 column，折下來，那我可以這樣做：

# In[83]:


air_quality_long = air_quality_wide.melt(
    id_vars=["city","country", "date.utc"],
    value_vars=["BETR801", "FR04014", "London Westminster"],
    var_name="location", # 轉成新column後的 column name
    value_name="NO_2", # 轉成新 column 後的 value name
)
air_quality_long


# ## 如何 concat (R 的 bind_rows, bind_cols)

# ### concat (axis = 0) (bind_rows)

# In[85]:


df1 = pd.DataFrame({
    "a": [1,2,3],
    "b": [4,5,6]
})
df2 = pd.DataFrame({
    "a": [7,8,9],
    "b": [10,11,12]
})

print(df1)
print(df2)


# In[86]:


pd.concat([df1, df2], axis = 0)


# ## 如何 merge (R 的 join)

# In[87]:


df1 = pd.DataFrame({
    "a": ["A", "A", "B"],
    "b": [4,5,6]
})
df2 = pd.DataFrame({
    "a": ["A","B","C"],
    "c": ["AA","BB","CC"]
})

print(df1)
print(df2)


# In[88]:


df1.merge(df2, how = "left", on = "a")


# * 如果要merge的名稱不同，例如這樣

# In[89]:


df1 = pd.DataFrame({
    "a1": ["A", "A", "B"],
    "b": [4,5,6]
})
df2 = pd.DataFrame({
    "a2": ["A","B","C"],
    "c": ["AA","BB","CC"]
})

print(df1)
print(df2)


# In[90]:


df1.merge(df2, how = "left", left_on = "a1", right_on = "a2")


# ## 如何處理 time-series data

# In[91]:


air_quality = pd.read_csv("data/air_quality_no2_long.csv")
air_quality = air_quality.rename(columns={"date.utc": "datetime"})
air_quality.head()


# * 我們首先看一下這筆資料，他的 `datetime` 欄位，是哪種 type:

# In[93]:


air_quality.info()


# * 可以發現， datetime 是 "object"，就是文字/類別的意思，所以我先把他轉為 datetime 格式

# In[94]:


air_quality["datetime"] = pd.to_datetime(air_quality["datetime"])
air_quality["datetime"]


# * 可以看到，現在 dtype 是 datetime64 了
# * 那為啥轉格式重要？因為有很多好用的 method 可以用

# ### 最大最小值

# * 就很直觀的，想看時間資料的最大值和最小值：

# In[95]:


print(air_quality["datetime"].min())
print(air_quality["datetime"].max())


# * 可以看到，時間最早是 5/7，最晚是 6/21. 
# * 那我還可以看一下時間距離多久？

# In[96]:


air_quality["datetime"].max() - air_quality["datetime"].min()


# ### 從時間資料中，擷取 年/月/日/星期幾...

# * 要擷取的這些資訊，都是 datetime 這個 series 的 attribute，我們可以這樣取

# In[154]:


print("datetime: ", air_quality["datetime"][0])
print("date: ", air_quality["datetime"].dt.date[0])
print("year: ", air_quality["datetime"].dt.year[0])
print("month: ", air_quality["datetime"].dt.month[0])
print("day: ", air_quality["datetime"].dt.day[0])
print("hour: ", air_quality["datetime"].dt.hour[0])
print("minute: ", air_quality["datetime"].dt.minute[0])
print("second: ", air_quality["datetime"].dt.second[0])
print("weekday: ", air_quality["datetime"].dt.weekday[0])


# * 可以看到，我取出 series 後，我還得用 `dt` 這個 accesor，他才知道我要調用 datetime 的 method，然後後面就直接用 date/year/month...等 attribute

# * 來練習一下吧，我如果想新增一個欄位，是只取出月份，那我可以這樣做：

# In[101]:


air_quality["month"] = air_quality["datetime"].dt.month # .dt 是調用datetime的 method/attribute，month看起來是attribute
air_quality.head()


# * 如果想取出星期幾(weekday)，我可以這樣做：

# In[102]:


air_quality["weekday"] = air_quality["datetime"].dt.weekday # .dt 是調用datetime的 method/attribute，month看起來是attribute
air_quality.head()


# * 我如果想看每個location，每個weekday，平均的 NO2 濃度，我就可以這樣做：

# In[103]:


air_quality.groupby(["location", "weekday"])["value"].mean()


# * 我想畫每個小時的平均 NO2 濃度

# In[118]:


fig, axs = plt.subplots(figsize=(12, 4))

(
    air_quality
        .assign(hour = lambda df: df.datetime.dt.hour)
        .groupby("hour")["value"]
        .mean()
        .plot(kind='bar', rot=0, ax=axs)
)


# ### slicing datetime

# * 我們這邊先把資料做成 wide 的形式：

# In[131]:


no_2 = air_quality.pivot(
    index = "datetime",
    columns = "location",
    values = "value"
).reset_index()
no_2.head()


# * 可以看到，現在有三條時間序列
# * 我如果想取出 "2019-05-20" ~ "2019-05-21" 的資料，我可以這樣做：

# In[137]:


no_2[(no_2.datetime >= "2019-05-20") & (no_2.datetime <= "2019-05-21")].head()


# * 但我還有另外一招，我可以把 datetime 挪去 index，然後直接篩選：

# In[136]:


no_2_with_datetime_index = no_2.set_index("datetime")
no_2_with_datetime_index["2019-05-20":"2019-05-21"].head()


# * 帥吧！接下來，我就可以畫出這三條時間序列：

# In[138]:


no_2_with_datetime_index["2019-05-20":"2019-05-21"].plot()


# ### Resample 成不同的 frequency

# * 其實 resample 的語法，就是 groupby 再 aggregate 的 shortcut. 
# * 舉個例子就懂了。
# * 我如果想看每天各個location的平均NO2的值 (所以把 ymd_hms 的 frequency 改成 ymd 而已)，那我得這樣做：

# In[151]:


(
    no_2
        .assign(Date = lambda df: df.datetime.dt.date)
        .groupby("Date")
        .mean()
        .plot(style="-o", figsize=(10, 5))
);


# * 那我現在可以用這個 resample 的語法，很快做到這件事

# In[149]:


(
    no_2
        .set_index("datetime")
        .resample("D")
        .mean()
        .plot(style="-o", figsize=(10, 5))
);


# ## 如何 manipulate textual data

# * 這一節要用的 data 是 Titanic

# In[152]:


titanic = pd.read_csv("data/titanic.csv")
titanic.head()


# ### 把某個欄位全轉小寫

# * 很快就可以聯想到 `str.lower()` 這個 method，所以作法就是：

# In[153]:


titanic["Name"].str.lower()


# ### 把某個欄位依照pattern切開

# * 舉例來說，`Name` 這個欄位，就用 `,` 來分隔出 first name 和 last name. 
# * 所以，我想把它切開來後，分別叫他 first name 和 last name

# In[156]:


titanic["split_res"] = titanic["Name"].str.split(",")
titanic[["Name", "split_res"]]


# * 那如果要再分成 first_name 和 last_name，就得這樣：

# In[159]:


titanic["first_name"] = titanic["split_res"].str.get(0) # 取第0個element
titanic["last_name"] = titanic["split_res"].str.get(1) # 取第0個element
titanic[["Name", "split_res", "first_name", "last_name"]]


# ### 是否包含某字串

# * 如果我想找找看，名字裡面，有出現 `Countess` 的人，那我可以這樣做：

# In[160]:


titanic[titanic["Name"].str.contains("Countess")]


# ### 字串長度

# * 我可以這樣，來造出字長：

# In[161]:


titanic["Name"].str.len()


# * 所以，我如果想找出名字最長的人，我可以這樣做：

# In[171]:


(
    titanic
        .assign(name_length = lambda df: df.Name.str.len())
        .sort_values("name_length", ascending=False)
        [["Name", "name_length"]]
        .head(3) # 前三名
)


# ### 字串取代

# * 我如果想把 `Sex` 這個欄位的 "female"，取代為 "F"，我可以這樣做:

# In[172]:


titanic["Sex"].str.replace("female", "F")


# * 所以，我如果想把 `Sex` 這個欄位的 "female" 改成 "F", "male" 改成 "M"，那我可以這樣做：

# In[173]:


titanic["Sex_short1"] = titanic["Sex"].str.replace("female", "F")
titanic["Sex_short1"] = titanic["Sex_short1"].str.replace("male", "M")
titanic[["Sex", "Sex_short1"]].head()


# * 事實上，如果你是要做這種取代的話，更好的做法是這樣：

# In[175]:


titanic["Sex_short2"] = titanic["Sex"].replace({"male": "M", "female": "F"})
titanic[["Sex","Sex_short1","Sex_short2"]].head()

