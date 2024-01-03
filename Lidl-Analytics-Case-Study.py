#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:purple;text-align:center">Lidl Analytics – Analytics Case Study</h1>

# ### Description and Objectives 
# + Lidl wants to expand its confectionery line but debates between a cookie-based or fruit gummy candy.
# + Market research was conducted to assess existing sweets popularity.
# + The task involves analyzing data to understand which candy traits drive popularity.
# + The goal is to recommend key characteristics for a successful new candy.

# ## Assumptions:
# + It's not clear if we prefer high or low sugar content in candies. High sugar can affect health, so knowing this would impact our choices.
# 
# + Since people in the survey didn't have to buy the candies, it seems like the price might not affect how well a candy does. But we should check more to be sure.
# 
# + We're not sure if the company wants candies with high or moderate win percentages. I assumed we aim for high win percentages because there are more candies with moderate wins.
# 
# + The data doesn't tell us which people we're targeting (like kids, young adults, or older people) or their ages. So, I assumed we're not focusing on any specific group for this analysis.

# ### import required libraries.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


url = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/candy-power-ranking/candy-data.csv'


# ### load  **Candy Power Ranking** dataset

# In[3]:


df=pd.read_csv(url)
df.head(5)


# ### inspect the dataset

# In[4]:


df.info()


# **The Candy Power Ranking** dataset contains 9 categorical variables **(chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer, hard, bar, pluribus)**, 3 continuous variables **(sugarpercent, pricepercent, winpercent)**, and 1 nominal variable **competitorname**.

# + The dataset contains the following fields:
# 
#    + **chocolate**	Does it contain chocolate?
#    + **fruity**	Is it fruit flavored?
#    + **caramel**	Is there caramel in the candy?
#    + **peanutalmondy**	Does it contain peanuts, peanut butter or almonds?
#    + **nougat**	Does it contain nougat?
#    + **crispedricewafer**	Does it contain crisped rice, wafers, or a cookie component?
#    + **hard**	Is it a hard candy?
#    + **bar**	Is it a candy bar?
#    + **pluribus**	Is it one of many candies in a bag or box?
#    + **sugarpercent**	The percentile of sugar it falls under within the data set.
#    + **pricepercent**	The unit price percentile compared to the rest of the set.
#    + **winpercent**	The overall win percentage 

# For categorical variables, **0** signifies **does not contain that property**,' while **1** indicates **contains that property**

# # Exploratory Data Analysis (EDA)

# ### check dimension of the dataset

# In[5]:


df.shape


# dataset contains 85 rows and 13 columns

# ### check for missing or null values

# In[6]:


df.isnull().sum()


# there is no **missing** or **null** value :)

# ### check for duplicate values

# In[7]:


df.duplicated().sum()


# the dataset is seemed to be already cleaned. so we do not need to clean the dataset.:)

# ### descriptive statistics on continuous variables

# In[8]:


df.describe(include='float64').T


# An interested observation is that both 'sugarpercent' and 'pricepercent' exhibit remarkably similar statistical characteristics, showcasing nearly identical values for mean, standard deviation, minimum, and maximum:

# ### descriptive statistics on categorical variables

# In[9]:


cols=df.columns.tolist() ## cols stores all the column names.


# In[10]:


df[[col for col in cols if col not in ['sugarpercent','pricepercent','winpercent',
                                       'competitorname']]].astype('object').describe().T 
## converting int->object in order to perform describe method on categorical variables


# Mejority of the candies do not feature **caramel**, **peanutyalmondy**, **nougat**, **crispedricewafer**.

# ### checking for **outlier** in **sugarpercent** and **pricepercent**

# In[11]:


data = {
    'sugarpercent': df['sugarpercent'],
    'pricepercent': df['pricepercent']
}
df_box = pd.DataFrame(data)
# Creating a horizontal boxplot for both variables
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_box, orient='h',palette='crest')
plt.title('Boxplot of Sugar Percent and Price Percent')
plt.xlabel('Percent')
plt.show()


# there is no **outlier**

# ### Let's examine the seven most popular and seven least popular candies.

# In[12]:


plt.figure(figsize=(10, 7))
sorted_df=df.sort_values(by='winpercent',ascending=False)
sns.barplot(data=pd.concat([sorted_df.head(7),pd.DataFrame({'competitorname':['(******)'],'winpercent':[0]}),
     sorted_df.tail(7)]),x='winpercent',y='competitorname',palette=['#00FFFF']*7 +['white']+ ['#FFD700']*7)
plt.xlabel('popularity in (%)')
plt.ylabel('Name of the Candy')
plt.yticks(rotation=30)
plt.title("Seven most popular and seven least popular candies based on win percent")
plt.show()


# + **Reese's Peanut Butter cup** is the most popular Candy with the winpercent of more than $80\%$.
# + **Nik L Nip** is the least popular Candy with the winpercent of almost $20\%$.

# ## Let's create visualizations highlighting the distinguishing characteristics that contribute to a candy's popularity, whether it ranks among the most favored or least favored.

# In[13]:


cat_vars=["chocolate","fruity","caramel","peanutyalmondy","nougat","crispedricewafer","hard","bar","pluribus"]
cont_vars=['sugarpercent','pricepercent','winpercent']


# In[14]:


# function to visualize the properties of  most and least 15 popular Candies.
def visualize_most_least_popular_candies(df_l,type):
    fig, ax = plt.subplots(4, 3, figsize=(13, 13))
    axes = ax.flatten()
    idx = 0
    for i in range(3):
        for j in range(3):  
            if idx < len(cat_vars):
                sns.countplot(x=cat_vars[idx], data=df_l, ax=ax[i, j], palette='crest')
                axes[idx].set_title(f'Is {cat_vars[idx]} ?')
                axes[idx].set_ylabel('Count')
                counts = df_l[cat_vars[idx]].value_counts(normalize=True) * 100
                no_percent = counts[0] if 0 in counts.index else 0
                yes_percent = counts[1] if 1 in counts.index else 0
                axes[idx].set_xticks([0, 1])
                axes[idx].set_xticklabels([f"No ({no_percent:.2f}%)", f"Yes ({yes_percent:.2f}%)"], rotation=45)
            else:
                # Hide the empty subplot
                ax[i, j].axis('off')
            idx += 1
    # Plot histograms for continuous variables
    sns.histplot(df_l['sugarpercent'], ax=ax[3, 0], kde=True, palette='crest')
    ax[3, 0].set_title('Sugar Percent Distribution')
    ax[3, 0].set_xlabel('Sugar Percent')

    sns.histplot(df_l['pricepercent'], ax=ax[3, 1], kde=True, palette='crest')
    ax[3, 1].set_title('Price Percent Distribution')
    ax[3, 1].set_xlabel('Price Percent')

    # Hide the empty subplot
    ax[3, 2].axis('off')

    plt.suptitle(f'Properties of {type} popular 15 Candies.')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# In[15]:


visualize_most_least_popular_candies(df.sort_values(by="winpercent",ascending=False).iloc[:15,:],"most")


# ### Analysis of Top 15 Candies:
#    + The majority $(93.3\%)$ of the most popular candies feature **chocolate**, while only a small proportion $(6.6\%)$ include **fruity** flavors.
#    + None of these top-ranked candies belong to the category of **hard** candies.
#    + Approximately half $(53.33\%)$ of them are not Candy **bars**.
#    + Almost $(53.33\%)$ of the most popular candies contains **peanutyalmondy**.
#    + Between $15\%$ to $30\%$ of these candies contain **pluribus**, **nougat**, **caramel** and **crispedricewafer**, showing a relatively limited presence of these particular ingredients among the selection.
#    + Additionally, these popular candies exhibit characteristics of high **sugar conten**t and a tendency towards      **higher pricing**.

# In[16]:


visualize_most_least_popular_candies(df.sort_values(by="winpercent",ascending=True).iloc[:15,:],"least")


# ### Analysis of Bottom 15 Candies:
#  + Examining the least popular 15 candies reveals stark differences compared to the top 15 candies.
# 
#  + Only a small percentage (6.6%) of the least popular candies feature chocolate, while a majority (53%) incorporate fruity flavors.
# 
#  + The majority of these less favored candies do not fall under the category of hard candies. None of them are Candy bars.
# 
# + Moreover, almost none of them contains **nougat**, **peanutyalmondy**, and **crispedricewafer**.
# 
# + Interestingly, these less popular candies tend to have lower sugar content and are priced relatively lower.

# **In summary**, popular candies often feature **chocolate**, **peanutyalmondy** exclude **hard** candies, and lean towards Candy **bars**. They may include certain ingredients but tend to have higher sugar content and pricing. On the other hand, less popular candies commonly incorporate fruity flavors, fewer Candy bar variations, and generally have lower sugar content and pricing.

# #### To facilitate broader analysis, let's transform the continuous variable 'winpercent' into a categorical variable with three classes: 'low winpercent,' 'medium winpercent,' and 'high winpercent.' This categorization will allow for a more generalized evaluation of the data.

# In[17]:


df['winpercent'].describe()


# In[18]:


# convert winpercent from continous->categorical variable
def convert_winpercent_to_cat(x):
    if x>=22.44 and x<43:
        return 'low'
    elif x>=43 and x<66:
        return 'medium'
    return 'high'


# In[19]:


df['winpercent_class']=df['winpercent'].apply(convert_winpercent_to_cat)


# ###  distribution of winpercent in three categories

# In[20]:


plt.figure(figsize=(7, 7))
ax=sns.countplot(x='winpercent_class', data=df, palette='crest')
plt.xticks(rotation=45)
plt.title("Distribution of winpercent in three categories")
plt.xlabel("Win percent")
ax.bar_label(ax.containers[0])
plt.show()


# + We notice that $44.7\%$ of the candies fall within the medium winpercent category, while $18.82\%$ belong to the high winpercent category.

# In[21]:



fig, ax = plt.subplots(4, 3, figsize=(18, 18))
axes = ax.flatten()
idx = 0

for i in range(3):
    for j in range(3):  
        if idx < len(cat_vars):
            sns.countplot(x=cat_vars[idx], data=df, ax=ax[i, j], palette='crest', hue='winpercent_class')
            axes[idx].set_title(f'Is {cat_vars[idx]} ?')
            axes[idx].set_ylabel('Count')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_xticklabels(['No','Yes'], rotation=45)
        else:
            # Hide the empty subplot
            ax[i, j].axis('off')
        idx += 1

# Plot separate histograms for continuous variables based on 'winpercent_class'
for cls, color in zip(['low', 'medium', 'high'], ['green', 'black', 'red']):
    sns.histplot(df[df['winpercent_class'] == cls]['sugarpercent'], ax=ax[3, 0], kde=True, color=color, label=cls.capitalize())
    sns.histplot(df[df['winpercent_class'] == cls]['pricepercent'], ax=ax[3, 1], kde=True, color=color, label=cls.capitalize())

ax[3, 0].set_title('Sugar Percent Distribution')
ax[3, 0].set_xlabel('Sugar Percent')
ax[3, 1].set_title('Price Percent Distribution')
ax[3, 1].set_xlabel('Price Percent')
ax[3, 0].legend()
ax[3, 1].legend()

# Hide the empty subplot
ax[3, 2].axis('off')

plt.suptitle('Analysis of Properties of all Candies based on winpercent class',fontsize=25,color='black')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ### Exploration of Candy Properties Across Three Categories:
# 
# 
#  + Candies exhibiting medium to high 'winpercent' values predominantly feature chocolate.
# 
#  + Candies with medium to low 'winpercent' commonly possess fruity characteristics.
# 
#  + Candies showing medium to high 'winpercent' tend to have the characteristic of being in a bar form.
# 
#  + Candies with high 'winpercent' often contain the 'peanutyalmondy' attribute.
# 
#  + Candies with lower 'winpercent' demonstrate lower sugar content and lower pricing. Conversely, candies with         medium to high 'winpercent' typically feature relatively higher sugar content and are priced higher.
# 
#  

# ### Observe correlation between each variable.

# In[22]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,fmt='0.1f',cmap='crest')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# We see that **chocolate** and **winpercent** have the largest correlation. Also bar and winpercent just as peanutyalmondy and winpercent have noticeable correlations. But we have to be careful because also **chocolate** and **bar** have a large correlation.

# In[23]:


# function which visualize the count of presence of a variable and its winpercent with bar and box plot
def show_count_and_winpercent_of_x_in_top20(variable):
    ig, ax = plt.subplots(1, 2, figsize=(10, 10))  # Corrected figsize and subplot layout

    data_top_20 = df[[variable, 'winpercent']].sort_values(by='winpercent', ascending=False).iloc[0:20,]

    sns.countplot(x=variable, data=data_top_20, palette='crest', ax=ax[0])  # Plotting count plot
    #sns.boxenplot(data=data_top_20,x='chocolate', y='winpercent', palette='crest', ax=ax[1])  # Plotting boxen plot
    sns.boxplot(data=data_top_20,x=variable,y='winpercent',palette='crest',orient='v')
    ax[0].set_title(f'Presence of {variable} in Top 20 Candies',fontsize=13,color='blue')  # Adding title to count plot
    ax[1].set_title(f'Winpercent Distribution for Top 20 Candies with {variable}',fontsize=13,color='blue')  # Adding title to boxen plot

    plt.tight_layout()  # Adjusts subplot parameters to give specified padding
    plt.show()
    


# ### chocolate vs fruity

# ### chocolate

# In[24]:


show_count_and_winpercent_of_x_in_top20('chocolate')


# ### fruity

# In[25]:


show_count_and_winpercent_of_x_in_top20('fruity')


# From top 20 candies only 2 of them are non-chocolate candies. Boxplot shows the strong correlation of **chocolate** to **winpercent**. we can  cookie-based candies are more popular than fruit-based candies.

# ## Analysis focusing on a single variable.

# ### bar

# In[26]:


chocolate_with_x('bar')


# In[ ]:


df[(df['bar']==1) & (df['chocolate']==0)]


# In[ ]:


df[(df['bar']==1) & (df['chocolate']==1)].shape


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(12, 6))  # Corrected figsize and subplot layout
sns.heatmap(df[['bar','winpercent']].corr(), ax=ax[0],annot=True,cmap='crest')  # Plotting count plot
sns.heatmap(df[df.chocolate == 1][['bar', 'winpercent']].corr(),annot=True, ax=ax[1],cmap='crest')  # Plotting boxen plot
sns.boxplot(data=df[df['chocolate']==1],x='bar',y='winpercent',palette='crest',orient='v')
ax[0].set_title('Overall',fontsize=13,color='blue')  # Adding title to count plot
ax[1].set_title('Only Chocolate',fontsize=13,color='blue')  # Adding title to boxen plot
ax[2].set_title('Only Chocolate',fontsize=13,color='blue')  # Adding title to boxen plot
plt.tight_layout()  # Adjusts subplot parameters to give specified padding
plt.show()


# Initially, a strong correlation was found between **bar** and **winpercent**, but this association was primarily driven by the presence of chocolate. Considering only candies with **chocolate**, the correlation between **bar** and **winpercent** vanished. There is also a small difference in winpercent between **chocolate** with **bar** or without **bar**.

# In[ ]:


## function to show a combination of chocolate with other variable.
def chocolate_with_x(x):
   return df[['chocolate', x, 'competitorname'
   ]].groupby(['chocolate', x], as_index=False).count()


# In[ ]:


## function show winpercent of a feature overall and in combination with chocolate with the help of bar and boxplot
def bar_box_plot_showing_winpercent(x):
    # Calculate the average winpercent for candies grouped by a given variable
    avg_winpercent = df.groupby(x)['winpercent'].mean().reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Corrected figsize and subplot layout

    sns.barplot(x=x,y='winpercent', data=avg_winpercent, palette='crest', ax=ax[0])  # Plotting count plot
    sns.boxplot(data=df[df['chocolate']==1], x=x, y="winpercent", palette="crest")

    ax[0].set_title('Overall',fontsize=13,color='blue')  # Adding title to count plot
    ax[1].set_title('Only Chocolate',fontsize=13,color='blue')  # Adding title to boxen plot
    ax[0].set_ylabel('Avg winpercent',fontsize=13)
    ax[1].set_ylabel('winpercent',fontsize=13)
    plt.tight_layout()  # Adjusts subplot parameters to give specified padding
    plt.show()
    plt.show()


# In[ ]:


def get_mean_winpercent_chocolate_with_x(x):
    return df[(df['chocolate']==1) & (df[x]==1)]['winpercent'].mean() 


# In[ ]:


def get_mean_winpercent_chocolate_without_x(x):
    return df[(df['chocolate']==1) & (df[x]==0)]['winpercent'].mean() 


# ### peanutyalmondy
# As we saw there is a strong correlation between peanutyalmondy and chocolate let's have a close look on it.

# In[ ]:


## combination of chocolate with peanutyalmondy
chocolate_with_x('peanutyalmondy')


# In[ ]:


get_mean_winpercent_chocolate_with_x('peanutyalmondy')


# In[ ]:


get_mean_winpercent_chocolate_without_x('peanutyalmondy')


# As candies that are **peanutyalmondy** nearly always contain **chocolate**, we narrow down our analysis to focus exclusively on candies that include **chocolate**.

# In[ ]:


## winpercent of peanutyalmondy overall and also in combination with chocolate
bar_box_plot_showing_winpercent('peanutyalmondy')


# The plot suggests that candies containing both **chocolate** and peanut/almond tend to outperform candies that have **chocolate** but lack peanut/almond.

# ### pluribus

# In[ ]:


## combination of chocolate with pluribus
chocolate_with_x('pluribus')


# In[ ]:


# avg winpercent by chocolate with pluribux
get_mean_winpercent_chocolate_with_x('pluribus')


# In[ ]:


# avg winpercent  by chocolate without pluribux
get_mean_winpercent_chocolate_without_x('pluribus')


# Interestingly, despite the presence of **chocolate** in 12 different candies that also contain the **pluribus** feature, the inclusion of **pluribus** alongside **chocolate** appears to result in a decrease in the **winpercent** of these candies.

# In[ ]:


## winpercent of pluribus overall and also in combination with chocolate
bar_box_plot_showing_winpercent('pluribus')


# candies with **pluribus** have in general low winpercent.

# ### nougat

# In[ ]:


## combination of chocolate with nougat
chocolate_with_x('nougat')


# In[ ]:


## winpercent of nougat overall and also in combination with chocolate
bar_box_plot_showing_winpercent('nougat')


# we can notice that candies containing **nougat**, on average and also with **chocolate** have higher winpercent.

# ### caramel

# In[ ]:


## combination of chocolate with nougat
chocolate_with_x('caramel')


# In[ ]:


get_mean_winpercent_chocolate_with_x('caramel')


# In[ ]:


get_mean_winpercent_chocolate_without_x('caramel')


# The majority of **chocolate** candies do not contain **caramel**. However, there are still some chocolate candies that include **caramel** as an ingredient. Additionally, inclusion of **caramel** with **chocolate** increases the avg **winpercent**

# In[ ]:


## winpercent of caramel overall and also in combination with chocolate
bar_box_plot_showing_winpercent('caramel')


# It's evident that candies containing **caramel**, on average, tend to exhibit slightly higher winpercent values.

# ### Fruity

# In[ ]:


## combination of chocolate with fruity
chocolate_with_x('fruity')


# In[ ]:


## winpercent of fruity overall and also in combination with chocolate
bar_box_plot_showing_winpercent('fruity')


# Like we already saw there is a negative correlation between **chocolate** and **fruity**. There is only one candy which has bothe **chocolate** and **fruity**.Most candies hava eithere **chocolate** or **fruity**. We also see the **winpercent** of Candies with **fruity** is lower than with **fruity**.

# ### Hard

# In[ ]:


## combination of chocolate with hard
chocolate_with_x('hard')


# Almost every **hard** cookie does not have **chocolate**

# In[ ]:


## winpercent of hard overall and also in combination with chocolate
bar_box_plot_showing_winpercent('hard')


# **Soft** candies generally exhibit higher winpercent values compared to **hard** candies.

# ### crispedricewafer

# In[ ]:


## combination of chocolate with crispedricewafer
chocolate_with_x('crispedricewafer')


# In[ ]:


get_mean_winpercent_chocolate_with_x('crispedricewafer')


# In[ ]:


get_mean_winpercent_chocolate_without_x('crispedricewafer')


# In[ ]:


## winpercent of crispedricewafer overall and also in combination with chocolate
bar_box_plot_showing_winpercent('crispedricewafer')


# candies with **crispedricewafer** get high winpercent overall as well as with **chocolate**

# ### pluribus

# In[ ]:


## combination of chocolate with pluribus
chocolate_with_x('pluribus')


# In[ ]:


## winpercent of pluribus overall and also in combination with chocolate
bar_box_plot_showing_winpercent('pluribus')


# Overall and also candies with **Chocolate** have low winpercent if the Candies from a bag or box. Although there are 12 candies in combination with **chocolate** and **pluribus**. we can say that the inclusion of **pluribus** with **chocolate** may decrease the win percent.

# ### pricepercent
# Since pricepercent differs for fruity and for chocolate, we separate the candies for the analysis.

# In[ ]:


chocolate_df = df[df['chocolate'] == 1]
fruity_df = df[df['fruity'] == 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Histogram (histplot) for pricepercent when chocolate
sns.histplot(data=chocolate_df, x='pricepercent',ax=axes[0])
axes[0].set_title('Histogram of Price Percent for Chocolate Candies')
axes[0].set_xlabel('Price Percent')
# Histogram (histplot) for pricepercent when fruity
sns.histplot(data=fruity_df, x='pricepercent',palette='crest',ax=axes[1])
axes[1].set_title('Histogram of Price Percent for Fruity Candies')
axes[1].set_xlabel('Price Percent')
plt.tight_layout()
plt.show()


# Candies containing **chocolate** tend to be slightly pricier compared to **fruity** candies. However, considering that the interviewees didn't have to purchase the candies, it's expected that the impact of price on preferences is minimal.

# ### sugarpercent
# Because of the association between sugar content and both fruity and chocolate candies, we distinguish between these two categories based on their sugar content.

# In[ ]:


chocolate_df = df[df['chocolate'] == 1]
fruity_df = df[df['fruity'] == 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Histogram (histplot) for pricepercent when chocolate
sns.histplot(data=chocolate_df, x='sugarpercent',palette='crest',ax=axes[0])
axes[0].set_title('Histogram of Sugar Percent for Chocolate Candies')
axes[0].set_xlabel('Sugar Percent')
# Histogram (histplot) for pricepercent when fruity
sns.histplot(data=fruity_df, x='sugarpercent',palette='crest',ax=axes[1])
axes[1].set_title('Histogram of Sugar Percent for Fruity Candies')
axes[1].set_xlabel('Sugar Percent')
plt.tight_layout()
plt.show()


# Candies containing **chocolate** generally exhibit a slight trend of having more sugar compared to **fruity** candies. However, this difference in sugar content appears to have a minimal impact or influence.

# ## Findings:
#  + **Chocolate's the Favorite:**
# 
#    + People love candies with chocolate the most.
#  + **Other Yummy Stuff Matters Too:**
# 
#     + Ingredients like peanuts, nougat, crispy wafer, and caramel also make candies more likable.
#  + Sweet Candies are a Hit:
# 
#    + People really enjoy candies with lots of sugar in them.
#  + **Price Doesn't Change Much:**
# 
#     + How much candies cost didn't affect how much people liked them in the survey.Hence, my assumption is that the price doesn't impact the winpercent
#  + **Best Mix: Chocolate and Peanuts Rock:**
# 
#    + Candies with chocolate and peanuts were liked more than candies with chocolate, nougat, crispy wafer, and    caramel.
#  + **Making Chocolate and Peanut Mix Even Better:**
# 
#     +  Now, let's try adding nougat, crispy wafer, and caramel to chocolate and peanuts to make candies even  tastier!
# 
# 
# 
# 
# 
# 

# ## ANOVA: Analysis of Variance

# In[32]:


from scipy import stats


# #### chocolate with peanutyalmondy

# In[41]:


peanutyalmondy_group=df[df['chocolate']==1][['peanutyalmondy','winpercent']].groupby('peanutyalmondy')
f_val,p_val=stats.f_oneway(peanutyalmondy_group.get_group(0)['winpercent'],peanutyalmondy_group.get_group(1)['winpercent'])
f_val,p_val


# With an F-value of approximately 7.3118 and a p-value of approximately 0.0105, the analysis suggests that there is a significant difference in 'winpercent' between the groups based on the presence or absence of 'peanutyalmondy' in chocolates that contain chocolate. The p-value being less than 0.05 indicates that there is strong evidence to reject the null hypothesis, suggesting that the 'peanutyalmondy' ingredient does have a significant impact on the 'winpercent' within chocolates that contain chocolate.

# #### chocolate and peanutyalmondy with caramel

# In[37]:


caramel_group=df[(df['chocolate']==1) & (df['peanutyalmondy']==1)][['caramel','winpercent']].groupby('caramel')
f_val,p_val=stats.f_oneway(caramel_group.get_group(0)['winpercent'],caramel_group.get_group(1)['winpercent'])
f_val,p_val


# With an F-value of approximately 0.5591 and a p-value of approximately 0.4718, it seems that there is not enough evidence to reject the null hypothesis. Therefore, based on this analysis, it appears that there is no significant difference in 'winpercent' between the groups with and without caramel in this dataset.

# #### chocolate and peanutyalmondy with caramel

# In[38]:


crispedricewafer_group=df[(df['chocolate']==1) & (df['peanutyalmondy']==1)][['crispedricewafer','winpercent']].groupby('crispedricewafer')
f_val,p_val=stats.f_oneway(crispedricewafer_group.get_group(0)['winpercent'],crispedricewafer_group.get_group(1)['winpercent'])
f_val,p_val


# With an F-value of approximately 0.7314 and a p-value of approximately 0.4124, the analysis suggests that there is no significant difference in 'winpercent' between the groups based on the presence or absence of 'crispedricewafer' in the dataset. These results indicate that, according to this analysis, the variable 'crispedricewafer' doesn't appear to have a significant impact on the 'winpercent' in the context of the chocolates with both chocolate and peanut/almond ingredients.

# In[39]:


nougat_group=df[(df['chocolate']==1) & (df['peanutyalmondy']==1)][['nougat','winpercent']].groupby('nougat')
f_val,p_val=stats.f_oneway(nougat_group.get_group(0)['winpercent'],nougat_group.get_group(1)['winpercent'])
f_val,p_val


# With an F-value of approximately 0.0547 and a p-value of approximately 0.8198, the analysis indicates that there is no significant difference in 'winpercent' between the groups based on the presence or absence of 'nougat' in the dataset. Therefore, according to this analysis, the 'nougat' ingredient does not seem to have a significant impact on the 'winpercent' within chocolates that contain both chocolate and peanut/almond ingredients.

# In[ ]:


df[df['chocolate']==1]['winpercent'].mean()


# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1)]['winpercent'].mean()


# **chocolate** in combination with **peanutyalmondy** has higher winpercent than **chocolate** alone

# In[ ]:


df[(df['chocolate']==1) & (df['crispedricewafer']==1)]['winpercent'].mean()


# **chocolate** in combination with **crispedricewafer** has higher winpercent than **chocolate** alone. but this is lower than **chocolate** in combination with **peanutyalmondy**

# ### chocolate, peanutyalmondy and crispedricewafer

# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) &  
   (df['crispedricewafer']==1)].shape


# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) &  
   (df['crispedricewafer']==1)]['winpercent'].mean()


# ### chocolate, peanutyalmondy and not crispedricewafer

# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) &  
   (df['crispedricewafer']==0)].shape


# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) & 
   (df['crispedricewafer']==0)]['winpercent'].mean()


# Among candies featuring chocolate, peanutyalmondy, and crispedricewafer, there's only one. Interestingly, by removing crispedricewafer from this combination of chocolate and peanutyalmondy, we notice an increase in the winpercent.

# ### chocolate, peanutyalmondy and not nougat

# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) &  
   (df['nougat']==0)].shape


# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) & 
   (df['nougat']==0)]['winpercent'].mean()


# ### chocolate, peanutyalmondy and nougat¶

# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) &  
   (df['nougat']==1)].shape


# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) & 
   (df['nougat']==1)]['winpercent'].mean()


# ### chocolate, peanutyalmondy, and caramel 

# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) & 
   (df['caramel']==1)].shape


# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) & 
   (df['caramel']==1)]['winpercent'].mean()


# there are only 3 Candies with **chocolate**, **peanutyalmondy**, and **caramel** and the winpercent is $64.37\%$.

# ## chocolate, peanutyalmondy, but not caramel 

# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) & 
   (df['caramel']==0)].shape


# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==1) & 
   (df['caramel']==0)]['winpercent'].mean()


# there are 9 Candies with chocolate, peanutyalmondy, and caramel and the winpercent is almost $70\%$.

# ### chocolate, not peanutyalmondy, and caramel

# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==0) & 
   (df['caramel']==1)].shape


# In[ ]:


df[(df['chocolate']==1) & (df['peanutyalmondy']==0) & 
   (df['caramel']==1)]['winpercent'].mean()


# there are 7 Candies with chocolate, peanutyalmondy, and caramel and the winpercent is almost  $67\%$.

# **In short**, candies with chocolate and peanutyalmondy (without caramel) achieve the highest winpercent. We'll examine if these candies significantly outperform others. Additionally, candies with chocolate and caramel (without peanut/almond) also perform remarkably well, significantly better than others. However, since there are only three candies with all three features, reaching statistical significance is not feasible, despite these being the most important features identified earlier.

# ## Utilize  machine learning models to gain deeper insights or a more comprehensive understanding.

# ### Data Preprocessing:

# Splitting into features and target variable

# In[ ]:


X = df[['chocolate', 'peanutyalmondy', 'crispedricewafer', 'nougat', 'caramel']] ## features
Y = df['winpercent'] # target variable


# 
# Given the absence of missing or duplicate values in our dataset, there's no immediate need for data cleaning procedures. Moreover, our categorical variables are already in a normalized form, which is advantageous. To enhance the accuracy of our model, we can apply the <code>StandardScaler</code> method specifically to the 'pricepercent' and 'sugarpercent' variables for scaling purposes.

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[['sugarpercent', 'pricepercent']] = scaler.fit_transform(X[['sugarpercent', 'pricepercent']])


# ### Split data into training and testing sets
#   I want to use $75\%$ data for training purpose and rest for the testing purpose.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.25, random_state=42)


# In[ ]:


X_train.shape, X_test.shape


# ### Model Selection

# Our objective is not centered around predicting **winpercent**; instead, we aim to ascertain the significance of each variable in relation to **winpercent**. Given the presence of numerous categorical variables in our dataset, I intend to utilize the <code>LinearRegression</code> model as a starting point.

# ### fit the model

# In[ ]:


from sklearn.linear_model import LinearRegression
# Initializing and fitting Logistic Regression model
logreg = LinearRegression()
logreg.fit(X_train, Y_train)


# ### model evaluation

# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predicting on the test set
y_pred = logreg.predict(X_test)

# Calculating R-squared
r_squared = r2_score(Y_test, y_pred)

# Calculating MSE and RMSE
mse = mean_squared_error(Y_test, y_pred)

print(f"R-squared: {r_squared}")
print(f"Mean Squared Error (MSE): {mse}")


# + R-squared (R² = 0.301):
# 
#    + The R-squared value measures the proportion of the variance in the dependent variable (target) that is predictable from the independent variables (features). In this case, around 30% of the variance in the target variable can be explained by the linear regression model. Higher values of R-squared closer to 1 indicate a better fit.
# +  Mean Squared Error (MSE = 136.65):
# 
#    + The MSE represents the average of the squared differences between predicted values and actual values. Here, the average squared difference between the predicted and actual values is approximately 136.65. Lower MSE values indicate a better fit.

# ### important features

# In[ ]:


feature_importance_df = pd.DataFrame({'Feature':X.columns
                                      , 'Importance': logreg.coef_})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(feature_importance_df['Feature'], feature_importance_df['Importance'],palette='crest')
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.xticks(rotation=60)
plt.show()


# Here we can still see that **chocolate** has the most importance following with **sugarpercent**, **pricepercent** and **peanutyalmondy**

# The **MSE** and **R-squared** values indicate that our current model performance is suboptimal. One possible reason for this could be the predominant use of categorical independent variables. Due to their categorical nature, determining a linear relationship becomes challenging. To address this limitation, let's explore the efficacy of employing tree-based models.

# ### convert **winpercent_class** into numberic

# In[ ]:


df['winpercent_class'].unique()


# In[ ]:


df['winpercent_class_numeric']=df['winpercent_class'].map({'low':0,'medium':1,'high':2})


# In[ ]:


X = df[['chocolate', 'peanutyalmondy', 'crispedricewafer', 'nougat', 'caramel']] ## features
Y = df['winpercent_class_numeric'] # target variable
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.25, random_state=42)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier on the training data
clf.fit(X_train, Y_train)
# Get feature importances
#importances = clf.feature_importances_


# ## model evaluation

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix
predictions = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(Y_test, predictions)
accuracy*100


# we can see that our model is performing well this time. it can predict 59% time correctly. 

# ### heatmap for confusion metrix

# In[ ]:


conf_matrix = confusion_matrix(Y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='crest', fmt='d', xticklabels=df['winpercent_class'].unique(), 
            yticklabels=df['winpercent_class'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=30)
plt.yticks(rotation=30)

plt.show()


# In[ ]:


from sklearn.metrics import classification_report

# Assuming predictions and y_test are your predicted labels and true labels respectively
# Replace predictions and y_test with your actual variables
report = classification_report(Y_test, predictions, target_names=df['winpercent_class'].unique())

# Print the classification report
print("Classification Report:")
print(report)


# In[ ]:


importances = clf.feature_importances_
feature_names = X.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(8, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()


# ## Summary

# The individual analysis indicates that **chocolate**, **peanutyalmondy**, **nougat**, **crispy wafer**,  and **caramel** emerge as the most influential features. Candies containing **chocolate** alongside either **peanutyalmondy** tend to be the top-performing ones.I haven't identified any additional ingredient that consistently contributes to high winpercent when paired with both **chocolate** and **peanutyalmondy**.
# 
# However, exploring these features together in multivariate analysis doesn't provide additional or more meaningful insights beyond what was revealed in the individual (univariate) analysis.

# ## Recommendation:
# Consider producing candies that include **chocolate** and **peanutyalmondy** ingredients, ideally with a moderate sugar content.
# 
