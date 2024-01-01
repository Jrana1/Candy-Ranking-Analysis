#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:blue;text-align:center">Fallstudie: Erweiterung des Süßwarensortiments<h1>

# ### import required libraries.

# In[72]:


import numpy as np
import pandas as pd


# ### import data set

# In[73]:


url = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/candy-power-ranking/candy-data.csv'


# In[74]:


df=pd.read_csv(url)
df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df[df['winpercent']>50].shape


# In[7]:


df.sort_values('winpercent',ascending=False).iloc[:10,]['sugarpercent'].mean()


# In[8]:


df.sort_values('winpercent',ascending=True).iloc[:10,]['sugarpercent'].mean()


# In[9]:


df


# In[10]:


df.sort_values('winpercent',ascending=False).iloc[:15,]#['pricepercent'].mean()


# In[11]:


df.sort_values('winpercent',ascending=True).iloc[:15,]#['pricepercent'].mean()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


props=["chocolate","fruity","caramel","peanutyalmondy","nougat","crispedricewafer","hard","bar","pluribus"]


# In[14]:


for col in props:
    print(f"{col}\n{df[col].value_counts()}")


# In[15]:


fig, ax = plt.subplots(3,3,figsize=(15,10))
idx=0
for i in range(3):
    for j in range(3):
        sns.countplot(x=props[idx],data=df, ax=ax[i,j])
        idx+=1


# In[16]:


fig, ax = plt.subplots(3,3,figsize=(15,10))
idx=0
for i in range(3):
    for j in range(3):
        sns.countplot(x=props[idx],data=df.sort_values('winpercent',ascending=False).iloc[:25,], ax=ax[i,j])
        idx+=1


# In[103]:


fig, ax = plt.subplots(3,3,figsize=(15,10))
idx=0
for i in range(3):
    for j in range(3):
        sns.countplot(x=props[idx],data=df.sort_values('winpercent',ascending=True).iloc[:15,], ax=ax[i,j])
        idx+=1


# In[18]:


df.head()


# In[19]:


sns.heatmap(df.corr(),annot=True,fmt='0.1f')


# We see that **chocolate** and **winpercent** have the largest correlation. Also bar and winpercent just as peanutyalmondy and winpercent have noticeable correlations. But we have to be careful because also **chocolate** and **bar** have a large correlation.

# In[20]:


df.corr()


# In[21]:


fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(df['pricepercent'])


# In[22]:


sns.regplot(df['pricepercent'],df['winpercent'])


# In[23]:


sns.regplot(df['pricepercent'],df['sugarpercent'])


# In[24]:


sns.regplot(df['winpercent'],df['sugarpercent'])


# In[25]:


df.head()


# In[47]:


fig, ax = plt.subplots(3, 3, figsize=(15, 10))
axes = ax.flatten()
idx = 0
for i in range(3):
    for j in range(3):
        if idx < len(props):
            sns.boxplot(x=props[idx], y='winpercent', data=df, ax=axes[idx])
            sns.swarmplot(x=props[idx], y='winpercent', data=df,ax=axes[idx],color="black")
            axes[idx].set_title(f'Boxplot of {props[idx]} against winpercent')
            axes[idx].set_xlabel(props[idx])  # Set the x-axis label
            axes[idx].set_ylabel('winpercent')  # Set the y-axis label
        else:
            axes[idx].axis('off')  # Hide any unused subplots
        idx += 1

plt.tight_layout()
plt.show()


# In[52]:


pivot_table = df.pivot_table(values='winpercent', index='chocolate', columns='fruity', aggfunc='mean')
pivot_table


# In[53]:


pivot_table = df.pivot_table(values='winpercent', index='chocolate', columns='fruity', aggfunc='mean')

# Create a heatmap to display the values of continuous_var across combinations of cat_var1 and cat_var2
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
plt.title('Heatmap showing the effect of two categorical variables on continuous variable')
plt.xlabel('chocolate')
plt.ylabel('fruity')
plt.show()


# In[56]:


for i in range(len(props)):
    for j in range(i + 1, len(props)):
        pivot_table = df.pivot_table(values='winpercent', index=props[i], columns=props[j], aggfunc='mean')
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
        plt.title(f'Heatmap: {props[i]} vs {props[j]} on Winpercent')
        plt.xlabel(props[j])
        plt.ylabel(props[i])
        plt.show()


# In[58]:


df.head()


# In[57]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[104]:


X=df.drop(['competitorname','winpercent','sugarpercent','pricepercent'],axis=1)
X.head()


# In[60]:


Y=df['winpercent']


# In[105]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100)


# In[65]:


X_train.shape


# In[66]:


X_test.shape


# In[106]:


lr=LinearRegression()


# In[107]:


lr.fit(X_train,Y_train)


# In[108]:


lr.intercept_


# In[109]:


lr.coef_


# In[110]:


y_pred = lr.predict(X_test)


# In[111]:


pd.DataFrame({'original':Y_test,'predicted':y_pred,'diff':Y_test-y_pred})


# In[112]:


lr.score(X_train,Y_train)


# In[113]:


lr.coef_


# In[114]:


plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns.tolist(), y=lr.coef_.tolist())
plt.title('Coefficients from Linear Regression Model')
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.xticks(rotation=60)
plt.show()


# In[89]:


fig, ax = plt.subplots(3, 3, figsize=(15, 10))
axes = ax.flatten()
idx = 0
for i in range(3):
    for j in range(3):
        if idx < len(props):
            sns.boxplot(x=props[idx], y='winpercent', data=df, ax=axes[idx])
            sns.swarmplot(x=props[idx], y='winpercent', data=df,ax=axes[idx],color="black")
            axes[idx].set_title(f'Boxplot of {props[idx]} against winpercent')
            axes[idx].set_xlabel(props[idx])  # Set the x-axis label
            axes[idx].set_ylabel('winpercent')  # Set the y-axis label
        else:
            axes[idx].aaxis('off')  # Hide any unused subplots
        idx += 1
plt.tight_layout()
plt.show()


# In[120]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, Y_train)

# Get feature importances from the trained model
feature_importances = rf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.xticks(rotation=60)
plt.show()

# Displaying feature importance table
print("Feature Importances:")
print(feature_importance_df)


# In[119]:


get_ipython().system('pip install xgboost')


# In[126]:


import xgboost as xgb
import matplotlib.pyplot as plt
# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=Y_train)

# Set XGBoost parameters
params = {'objective': 'reg:squarederror', 'random_state': 42}

# Train the XGBoost model
num_round = 100
model = xgb.train(params, dtrain, num_round)

# Get feature importances from the trained model
importance = model.get_score(importance_type='gain')
feature_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in XGBoost Model')
plt.xticks(rotation=60)
plt.show()

# Displaying feature importance table
print("Feature Importances:")
print(feature_importance_df)


# In[127]:


X=df.drop(['competitorname','winpercent'],axis=1)


# In[ ]:





# In[129]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100)


# In[130]:


import xgboost as xgb
import matplotlib.pyplot as plt
# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=Y_train)

# Set XGBoost parameters
params = {'objective': 'reg:squarederror', 'random_state': 42}

# Train the XGBoost model
num_round = 100
model = xgb.train(params, dtrain, num_round)

# Get feature importances from the trained model
importance = model.get_score(importance_type='gain')
feature_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in XGBoost Model')
plt.xticks(rotation=60)
plt.show()

# Displaying feature importance table
print("Feature Importances:")
print(feature_importance_df)


# In[131]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, Y_train)

# Get feature importances from the trained model
feature_importances = rf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.xticks(rotation=60)
plt.show()

# Displaying feature importance table
print("Feature Importances:")
print(feature_importance_df)


# In[132]:


df.head()


# In[133]:


df['sugarpercent'].describe()


# In[134]:


df['pricepercent'].describe()


# In[138]:


def fun1(x):
    if x>=0 and x<0.255:
        return 1
    elif x>=0.255 and x<0.651:
        return 2
    else:
        return 3


# In[143]:


df['pricepercent']=df['pricepercent'].apply(fun1)


# In[141]:


def fun2(x):
    if x>=0 and x<0.220000:
        return 1
    elif x>=0.220000 and x<0.465000:
        return 2
    else:
        return 3


# In[144]:


df['sugarpercent']=df['sugarpercent'].apply(fun1)


# In[ ]:


X=df.drop(['competitorname','winpercent'],axis=1)


# In[145]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100)


# In[149]:


import xgboost as xgb
import matplotlib.pyplot as plt
# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=Y_train)

# Set XGBoost parameters
params = {'objective': 'reg:squarederror', 'random_state': 42}

# Train the XGBoost model
num_round = 100
model = xgb.train(params, dtrain, num_round)

# Get feature importances from the trained model
importance = model.get_score(importance_type='gain')
feature_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in XGBoost Model')
plt.xticks(rotation=60)
plt.show()

# Displaying feature importance table
print("Feature Importances:")
print(feature_importance_df)


# In[147]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, Y_train)

# Get feature importances from the trained model
feature_importances = rf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.xticks(rotation=60)
plt.show()

# Displaying feature importance table
print("Feature Importances:")
print(feature_importance_df)


# In[6]:


df[['sugarpercent','pricepercent','winpercent']].describe()


# In[7]:


def fun1(x):
    if x>=0 and x<0.220000:
        return 1
    elif x>=0.220000 and x<0.732000:
        return 2
    return 3


# In[8]:


def fun2(x):
    if x>=0 and x<0.255000:
        return 1
    elif x>=0.255000 and x<0.651000:
        return 2
    return 3


# In[25]:


def fun3(x):
    if x>=0 and x<39.141056:
        return 1
    elif x>=39.141056 and x<59.863998:
        return 2
    return 3


# In[60]:


#df['sugarpercent']=df['sugarpercent'].apply(fun1)
#df['pricepercent']=df['pricepercent'].apply(fun2)
df['winpercent']=df['winpercent'].apply(fun3)


# In[11]:


df.head()


# In[12]:


import seaborn as sns


# In[13]:


sns.countplot('sugarpercent',data=df)


# In[14]:


sns.countplot('pricepercent',data=df)


# In[55]:


sns.countplot('winpercent',data=df)


# In[56]:


df['winpercent'].value_counts()


# In[32]:


df.head()


# In[36]:


sns.countplot('chocolate',data=df[df['winpercent']==2])


# In[37]:


sns.countplot('chocolate',data=df[df['winpercent']==1])


# In[38]:


sns.countplot('chocolate',data=df[df['winpercent']==3])


# In[43]:


import matplotlib.pyplot as plt


# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `props` contains the list of properties and `df` is your DataFrame

# Set a color palette for winpercent
winpercent_palette = "rocket_r"

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
idx = 0
for i in range(3):
    for j in range(3):
        sns.countplot(x=props[idx], data=df, hue='winpercent', ax=ax[i, j], palette=winpercent_palette)
        idx += 1

plt.tight_layout()
plt.show()


# In[66]:



sns.boxplot(x='winpercent', y='sugarpercent', data=df,palette='rocket_r')


# In[64]:


sns.boxplot(x='winpercent', y='pricepercent', data=df)


# In[75]:


import seaborn as sns


# In[76]:


sns.scatterplot(data=df,y='sugarpercent',x='winpercent',size='pricepercent')

