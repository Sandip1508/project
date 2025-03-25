#!/usr/bin/env python
# coding: utf-8

# ## Name:Sonawane Sandip Anil 
# ###### A/P Dhandarne, Tal-Shindkheda, Dist-Dhule, 425406
# ###### Contact: +91-9579329567
# ###### Email: sandip15082003@gmail.com
# ###### Date of Birth: 15 August 2003 
# 

# # YouTube Trending Videos Analysis (More Than 40,000 Videos) 

# ### Introduction

# YouTube is the most popular and most used video platfrom in the world today. YouTube has a list of trending videos that is updated constantly. Here we will use **Python** with some packages like **Pandas** and **Matplotlib** to analyze a dataset that was collected over 205 days. For each of those days, the dataset contains data about the trending videos of that day. It contains data about **more than 40,000 trending videos**. We will analyze this data to get insights into YouTube trending videos, to see what is common between these videos. Those insights might also be used by people who want to increase popularity of their videos on YouTube.
# 
# The dataset that we will use is obtained from Kaggle here. It contains data about trending videos for many countries. Here we will analyze USA trending videos

# ### Goals of the analysis

# We want to answer questions like:
# 
#   * How many views do our trending videos have? Do most of them have a large number of views? Is having a large number of 
#     views required for a video to become trending?
# 
#   * The same questions above, but applied to likes and comment count instead of views.
# 
#   * Which video remained the most on the trendin-videos list?
#     
#   * How many trending videos contain a fully-capitalized word in their titles?
#     
#   * What are the lengths of trending video titles? Is this length related to the video becoming trendy?
#     
#   * How are views, likes, dislikes, comment count, title length, and other attributes correlate with (relate to) each other?
#     How are they connected?
#     
#   * What are the most common words in trending video titles?
#     
#   * Which YouTube channels have the largest number of trending videos?
#     
#   * Which video category (e.g. Entertainment, Gaming, Comedy, etc.) has the largest number of trending videos?
#     
#   * When were trending videos published? On which days of the week? at which times of the day?

# ### Table of contents

# * Importing some packages
# * Reading the dataset
# * Getting a feel of the dataset
# * Data cleaning
# * Dataset collection years
# * Describtion of numerical columns
#      * Views histogram
#      * Likes histogram
#      * Comment count histogram
# * Description on non-numerical columns
# * How many trending video titles contain capitalized word?
# * Video title lengths
# * Correlation between dataset variables
# * Most common words in video titles
# * Which channels have the largest number of trending videos?
# * Which video category has the largest number of trending videos?
# * Trending videos and their publishing time
# * How many trending videos have an error?
# * How many trending videos have their commets disabled?
# * How many trending videos have their ratings disabled?
# * How many videos have both comments and ratings disabled?
# * Conclusions 
# 
# 
# ##### Let's get started

# ##  Importing some packages ##

# First, we import some Python packages that will help us analyzing the data, especially pandas for data analysis and matplotlib for visualization

# In[6]:


import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
from collections import Counter
import datetime
import wordcloud
import json


# In[7]:


# Hiding warnings for cleaner display
warnings.filterwarnings('ignore')

# Configuring some options
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# If you want interactive plots, uncomment the next line
# %matplotlib notebook


# ### Reading the dataset

# Then we read the dataset file which is in csv format

# In[8]:


df = pd.read_csv("USvideos.csv")


# We set some configuration options just for improving visualization graphs; nothing crucial

# In[9]:


PLOT_COLORS = ["#268bd2", "#0052CC", "#FF5722", "#b58900", "#003f5c"]
pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('font', family='Arial', weight='400', size=10)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)


# ### Getting a feel of the dataset

# Let's get a feel of our dataset by displaying its first few rows

# In[10]:


df.head()


# Now, let's see some information about our dataset

# In[11]:


df.info()


# We can see that there are 40,949 entries in the dataset. We can see also that all columns in the dataset are complete (i.e. they have 40,949 non-null entries) except description column which has some null values; it only has 40,379 non-null values.

# ### Data cleaning 

# The description column has some null values. These are some of the rows whose description values are null. We can see that null values are denoted by NaN

# In[12]:


df[df["description"].apply(lambda x: pd.isna(x))].head(3)


# So to do some sort of data cleaning, and to get rid of those null values, we put an empty string in place of each null value in the description column

# In[13]:


df["description"] = df["description"].fillna(value="")


# ### Dataset collection years

# Let's see in which years the dataset was collected

# In[14]:


cdf = df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts() \
            .to_frame().reset_index() \
            .rename(columns={"index": "year", "trending_date": "No_of_videos"})

fig, ax = plt.subplots()
_ = sns.barplot(x="year", y="No_of_videos", data=cdf, palette=sns.color_palette(['#ff764a', '#ffa600'], n_colors=7), ax=ax)
_ = ax.set(xlabel="Year", ylabel="No. of videos")


# In[15]:


df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts(normalize=True)


# We can see that the dataset was collected in 2017 and 2018 with 77% of it in 2018 and 23% in 2017.

# ###  Describtion of numerical columns

# Now, let's see some statistical information about the numerical columns of our dataset

# In[16]:


df.describe()


# We note from the table above that
# 
#    * The average number of views of a trending video is 2,360,784. The median value for the number of views is 681,861, which  means that half the trending videos have views that are less than that number, and the other half have views larger than that number
#    * The average number of likes of a trending video is 74,266, while the average number of dislikes is 3,711. The
#    * Average comment count is 8,446 while the median is 1,856
# 
# How useful are the observations above? Do they really represent the data? Let's examine more.

# #### Views histogram

# let's plot a __histogram__ for the views column to take a look at its distribution: to see how many videos have between 10 million and 20 million views, how many videos have between 20 million and 30 million views, and so on.

# In[17]:


fig, ax = plt.subplots()
_ = sns.distplot(df["views"], kde=False, color=PLOT_COLORS[4], 
                 hist_kws={'alpha': 1}, bins=np.linspace(0, 2.3e8, 47), ax=ax)
_ = ax.set(xlabel="Views", ylabel="No. of videos", xticks=np.arange(0, 2.4e8, 1e7))
_ = ax.set_xlim(right=2.5e8)
_ = plt.xticks(rotation=90)


# We note that the vast majority of trending videos have 5 million views or less. We get the 5 million number by calculating
# 
#    ##                                      0.1×10*8/2=5×10*6
#  
# 
# Now let us plot the histogram just for videos with 25 million views or less to get a closer look at the distribution of the data

# In[18]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["views"] < 25e6]["views"], kde=False,color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Views", ylabel="No. of videos")


# Now we see that the majority of trending videos have 1 million views or less. Let's see the exact percentage of videos less than 1 million views

# In[19]:


df[df['views'] < 1e6]['views'].count() / df['views'].count() * 100


# So, it is around 60%. Similarly, we can see that the percentage of videos with less than 1.5 million views is around 71%, and that the percentage of videos with less than 5 million views is around 91%.
# 
# #### Likes histogram
# After views, we plot the histogram for likes column

# In[20]:


plt.rc('figure.subplot', wspace=0.9)
fig, ax = plt.subplots()
_ = sns.distplot(df["likes"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, 
                 bins=np.linspace(0, 6e6, 61), ax=ax)
_ = ax.set(xlabel="Likes", ylabel="No. of videos")
_ = plt.xticks(rotation=90)


# We note that the vast majority of trending videos have between 0 and 100,000 likes. Let us plot the histogram just for videos with 1000,000 likes or less to get a closer look at the distribution of the data

# In[21]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["likes"] <= 1e5]["likes"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Likes", ylabel="No. of videos")


# Now we can see that the majority of trending videos have 40000 likes or less with a peak for videos with 2000 likes or less.
# 
# Let's see the exact percentage of videos with less than 40000 likes

# In[22]:


df[df['likes'] < 4e4]['likes'].count() / df['likes'].count() * 100


# Similarly, we can see that the percentage of videos with less than 100,000 likes is around 84%
# 
# #### Comment count histogram

# In[23]:


fig, ax = plt.subplots()
_ = sns.distplot(df["comment_count"], kde=False, rug=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Comment Count", ylabel="No. of videos")


# Let's get a closer look by eliminating entries with comment count larger than 200000 comment

# In[24]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["comment_count"] < 200000]["comment_count"], kde=False, rug=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, 
                 bins=np.linspace(0, 2e5, 49), ax=ax)
_ = ax.set(xlabel="Comment Count", ylabel="No. of videos")


# We see that most trending videos have around
# 
# #### 25000/6≈4166 comments
#  
# 
# since each division in the graph has six histogram bins.
# 
# As with views and likes, let's see the exact percentage of videos with less than 4000 comments

# In[25]:


df[df['comment_count'] < 4000]['comment_count'].count() / df['comment_count'].count() * 100


# In a similar way, we can see that the percentage of videos with less than 25,000 comments is around 93%.
# 
# ### Description on non-numerical columns
# After we described numerical columns previously, we now describe non-numerical columns

# In[26]:


df.describe(include = ['O'])


# From the table above, we can see that there are 205 unique dates, which means that our dataset contains collected data about trending videos over **205** days.
# 
# From video_id description, we can see that there are 40949 videos (which is expected because our dataset contains 40949 entries), but we can see also that there are only 6351 unique videos which means that some videos appeared on the trending videos list **on more than one day**. The table also tells us that the top frequent title is WE MADE OUR MOM CRY...HER DREAM CAME TRUE! and that it appeared 30 times on the trending videos list.
# 
# But there is something strange in the description table above: Because there are 6351 unique video IDs, we expect to have 6351 unique video titles also, because we assume that each ID is linked to a corresponding title. One possible interpretation is that a trending video had some title when it appeared on the trending list, then it appeared again on another day but with a modified title. Similar explaination applies for description column as well. For publish_time column, the unique values are less than 6351, but there is nothing strange here, because two different videos may be published at the same time.
# 
# To verify our interpretation for title column, let's take a look at an example where a trending video appeared more than once on the trending list but with different titles

# In[27]:


grouped = df.groupby("video_id")
groups = []
wanted_groups = []
for key, item in grouped:
    groups.append(grouped.get_group(key))

for g in groups:
    if len(g['title'].unique()) != 1:
        wanted_groups.append(g)

wanted_groups[0]


# We can see that this video appeared on the list with two different titles.
# 
# ### How many trending video titles contain capitalized word?
# Now we want to see how many trending video titles contain at least a capitalized word (e.g. HOW). To do that, we will add a new variable (column) to the dataset whose value is True if the video title has at least a capitalized word in it, and False otherwise

# In[28]:


def contains_capitalized_word(s):
    for w in s.split():
        if w.isupper():
            return True
    return False


df["contains_capitalized"] = df["title"].apply(contains_capitalized_word)

value_counts = df["contains_capitalized"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'}, startangle=45)
_ = ax.axis('equal')
_ = ax.set_title('Title Contains Capitalized Word?')


# In[29]:


df["contains_capitalized"].value_counts(normalize=True)


# We can see that 44% of trending video titles contain at least a capitalized word. We will later use this added new column contains_capitalized in analyzing correlation between variables.
# 
# ### Video title lengths
# Let's add another column to our dataset to represent the length of each video title, then plot the histogram of title length to get an idea about the lengths of trnding video titles

# In[30]:


df["title_length"] = df["title"].apply(lambda x: len(x))

fig, ax = plt.subplots()
_ = sns.distplot(df["title_length"], kde=False, rug=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Title Length", ylabel="No. of videos", xticks=range(0, 110, 10))


# We can see that title-length distribution resembles a normal distribution, where most videos have title lengths between 30 and 60 character approximately.
# 
# Now let's draw a scatter plot between title length and number of views to see the relationship between these two variables

# In[31]:


fig, ax = plt.subplots()
_ = ax.scatter(x=df['views'], y=df['title_length'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Views", ylabel="Title Length")


# By looking at the scatter plot, we can say that there is no relationship between the title length and the number of views. However, we notice an interesting thing: videos that have 100,000,000 views and more have title length between 33 and 55 characters approximately.
# 
# ### Correlation between dataset variables
# Now let's see how the dataset variables are correlated with each other: for example, we would like to see how views and likes are correlated, meaning do views and likes increase and decrease together (positive correlation)? Does one of them increase when the other decrease and vice versa (negative correlation)? Or are they not correlated?
# 
# Correlation is represented as a value between -1 and +1 where +1 denotes the highest positive correlation, -1 denotes the highest negative correlation, and 0 denotes that there is no correlation.
# 
# Let's see the correlation table between our dataset variables (numerical and boolean variables only)

# In[32]:


df.corr()


# We see for example that views and likes are highly positively correlated with a correlation value of 0.85; we see also a high positive correlation (0.80) between likes and comment count, and between dislikes and comment count (0.70).
# 
# There is some positive correlation between views and dislikes, between views and comment count, between likes and dislikes.
# 
# Now let's visualize the correlation table above using a heatmap

# In[33]:


h_labels = [x.replace('_', ' ').title() for x in 
            list(df.select_dtypes(include=['number', 'bool']).columns.values)]

fig, ax = plt.subplots(figsize=(10,6))
_ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)


# The correlation map and correlation table above say that views and likes are highly positively correlated. Let's verify that by plotting a scatter plot between views and likes to visualize the relationship between these variables

# In[34]:


fig, ax = plt.subplots()
_ = plt.scatter(x=df['views'], y=df['likes'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Views", ylabel="Likes")


# We see that views and likes are truly positively correlated: as one increases, the other increases too—mostly.
# 
# Another verification of the correlation matrix and map is the scatter plot we drew above between views and title length as it shows that there is no correlation between them.

# In[35]:


title_words = list(df["title"].apply(lambda x: x.split()))
title_words = [x for y in title_words for x in y]
Counter(title_words).most_common(25)


# In[36]:


# wc = wordcloud.WordCloud(width=1200, height=600, collocations=False, stopwords=None, background_color="white", colormap="tab20b").generate_from_frequencies(dict(Counter(title_words).most_common(150)))
wc = wordcloud.WordCloud(width=1200, height=500,collocations=False, background_color="white",colormap="tab20b").generate(" ".join(title_words))
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation='bilinear')
_ = plt.axis("off")


# ### Which channels have the largest number of trending videos?

# In[37]:


cdf = df.groupby("channel_title").size().reset_index(name="video_count") \
    .sort_values("video_count", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8,8))
_ = sns.barplot(x="video_count", y="channel_title", data=cdf,
                palette=sns.cubehelix_palette(n_colors=20, reverse=True), ax=ax)
_ = ax.set(xlabel="No. of videos", ylabel="Channel")


# ### Which video category has the largest number of trending videos?
# First, we will add a column that contains category names based on the values in category_id column. We will use a category JSON file provided with the dataset which contains information about each category.

# In[38]:


with open("US_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df['category_name'] = df['category_id'].map(cat_dict)


# Now we can see which category had the largest number of trending videos

# In[39]:


cdf = df["category_name"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "category_name", "category_name": "No_of_videos"}, inplace=True)
fig, ax = plt.subplots()
_ = sns.barplot(x="category_name", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=16, reverse=True), ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
_ = ax.set(xlabel="Category", ylabel="No. of videos")


# We see that the Entertainment category contains the largest number of trending videos among other categories: around 10,000 videos, followed by Music category with around 6,200 videos, followed by Howto & Style category with around 4,100 videos, and so on.
# 
# ### Trending videos and their publishing time
# An example value of the publish_time column in our dataset is 2017-11-13T17:13:01.000Z. And according to information on this page: https://www.w3.org/TR/NOTE-datetime, this means that the date of publishing the video is 2017-11-13 and the time is 17:13:01 in Coordinated Universal Time (UTC) time zone.
# 
# Let's add two columns to represent the date and hour of publishing each video, then delete the original publish_time column because we will not need it anymore

# In[40]:


df["publishing_day"] = df["publish_time"].apply(
    lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d").date().strftime('%a'))
df["publishing_hour"] = df["publish_time"].apply(lambda x: x[11:13])
df.drop(labels='publish_time', axis=1, inplace=True)


# Now we can see which days of the week had the largest numbers of trending videos

# In[41]:


cdf = df["publishing_day"].value_counts()\
        .to_frame().reset_index().rename(columns={"index": "publishing_day", "publishing_day": "No_of_videos"})
fig, ax = plt.subplots()
_ = sns.barplot(x="publishing_day", y="No_of_videos", data=cdf, 
                palette=sns.color_palette(['#003f5c', '#374c80', '#7a5195', 
                                           '#bc5090', '#ef5675', '#ff764a', '#ffa600'], n_colors=7), ax=ax)
_ = ax.set(xlabel="Publishing Day", ylabel="No. of videos")


# We can see that the number of trending videos published on Sunday and Saturday are noticeably less than the number of trending videos published on other days of the week.
# 
# Now let's use publishing_hour column to see which publishing hours had the largest number of trending videos

# In[42]:


cdf = df["publishing_hour"].value_counts().to_frame().reset_index()\
        .rename(columns={"index": "publishing_hour", "publishing_hour": "No_of_videos"})
fig, ax = plt.subplots()
_ = sns.barplot(x="publishing_hour", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=24), ax=ax)
_ = ax.set(xlabel="Publishing Hour", ylabel="No. of videos")


# We can see that the period between 2PM and 7PM, peaking between 4PM and 5PM, had the largest number of trending videos. We notice also that the period between 12AM and 1PM has the smallest number of trending videos. But why is that? Is it because people publish a lot more videos between 2PM and 7PM? Is it because how YouTube algorithm chooses trending videos?
# 
# ### How many trending videos have an error?
# To see how many trending videos got removed or had some error, we can use video_error_or_removed column in the dataset

# In[43]:


value_counts = df["video_error_or_removed"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
        colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Video Error or Removed?')


# In[44]:


df["video_error_or_removed"].value_counts()


# We can see that out of videos that appeared on trending list (40949 videos), there is a tiny portion (23 videos) with errors.
# 
# ### How many trending videos have their commets disabled?
# To know that, we use comments_disabled column

# In[45]:


value_counts = df["comments_disabled"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie(x=[value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Comments Disabled?')


# In[46]:


df["comments_disabled"].value_counts(normalize=True)


# We see that only 2% of trending videos prevented users from commenting.
# 
# 
# 
# ### How many trending videos have their ratings disabled?
# To know that, we use ratings_disabled column

# In[47]:


value_counts = df["ratings_disabled"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
            colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Ratings Disabled?')


# In[48]:


df["ratings_disabled"].value_counts()


# We see that only 169 trending videos out of 40949 prevented users from commenting.
# 
# ### How many videos have both comments and ratings disabled?

# In[49]:


len(df[(df["comments_disabled"] == True) & (df["ratings_disabled"] == True)].index)


# So there are just 106 trending videos that have both comments and ratings disabled
# 
# ### Conclusions
# Here are the some of the results we extracted from the analysis:
# 
#    * We analyzed a dataset that contains information about YouTube trending videos for 205 days. The dataset was collected in 2017 and 2018. It contains **40949** video entry.
#    * 71% of trending videos have less than 1.5 million views, and **91%** have less than **5** million views.
#    * 68% of trending videos have less than 40,000 likes, and **84%** have less than **100,000** likes.
#    * 67% of trending videos have less than 4,000 comments, and **93%** have less than **25,000** comments.
#    * Some videos may appear on the trending videos list on more than one day. Our dataset contains 40494 entries but not for 40494 unique videos but for 6351unique videos.
#    * Trending videos that have **100,000,000** views and more have title length between 33 and 55 characters approximately.
#    * The delimiters - and | were common in trending video titles.
#    * The words 'Official', 'Video', 'Trailer', 'How', and '2018' were common also in trending video titles.
#    * There is a strong positive correlation between the number of views and the number of likes of trending videos: As one of them increases, the other increases, and vice versa.
#    * There is a strong positive correlation also between the number of likes and the number of comments, and a slightly weaker one between the number of dislikes and the number of comments.
#    * The category that has the largest number of trending videos is **'Entertainment'** with 9,964 videos, followed by 'Music' category with 6,472 videos, followed by 'Howto & Style' category with 4146 videos.
#    * On the opposite side, the category that has the smallest number of trending videos is 'Shows' with 57 videos, followed by 'Nonprofits & Activisim' with 57 videos, followed by 'Autos & Vehicles' with 384 videos.
# 
# 

# In[ ]:




