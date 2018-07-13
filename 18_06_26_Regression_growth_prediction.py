
# coding: utf-8

# # Predicting Economic growth
# by Daniel Kwasnitschka and Martin Siebert
# 
# 

# ### 1 Setting up the environment

# In[1]:


# import dependencies
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

# set options for displaying
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_rows', 500)
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 


# In[2]:


# Load Data exported from R into a DataFrame
df = pd.read_csv("GrowthData_Full.csv", sep=",", index_col=0)


# In[3]:


# Load list of all OECD countries with their wbcode
oecd = pd.read_excel("OECD_countries.xlsx")


# ### 2. Get to know the dataset

# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


oecd.head()


# In[7]:


df.describe()


# In[8]:


# distribution of GDPpcGrowthMA
sns.distplot(df['GDPpcGrowthMA'] , fit=norm);

(mu, sigma) = norm.fit(df['GDPpcGrowthMA'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('GDPpcGrowthMA distribution')


# ##### Feature Correlation

# In[9]:


# Heat Map for some Lag Variables to gain better understanding of their dependencies
df_lag = df.filter(regex='Lag')

df_lag["GDPpcGrowthMA"] = df["GDPpcGrowthMA"]

corrmat = df_lag[df_lag.columns[-20:]].corr()
f, ax = plt.subplots(figsize=(30, 20))
sns.set(font_scale=1.25)
sns.heatmap(corrmat, vmax=.8, square=True,annot=True, fmt='.2f', annot_kws={'size': 18});


# ### 3. Preprocessing

# ##### Missing Data Handling

# In[10]:


# checking for missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# display missing data TOP 5
missing_data.head()


# In[11]:


# handle missing data
# drop columns all features with missing values since only Lag Variables will be used for prediction
df = df.drop(["Recession","Recession5", "System","GDPpcGrowth"], axis=1)


# In[12]:


# checking for missing data after handling
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# display missing data TOP 5
missing_data.head()


# ##### Get Lag Variables

# In[13]:


# array with all Lag Variables
lag_cols = [col for col in df.columns if 'Lag' in col]


# In[14]:


print(lag_cols)


# In[15]:


# create df for training the models
df_lag = df.filter(regex='Lag')
df_lag["wbcode"] = df["wbcode"]
df_lag["GDPpcGrowthMA"] = df["GDPpcGrowthMA"]


# ##### Filter for OECD countries

# In[16]:


# filter to only include OECD countries
df_lag = df_lag.loc[df_lag["wbcode"].isin(oecd["wbcode"])]


# ##### Split into train and test set

# In[17]:


# create dummy variables for categorical features
df_lag = pd.get_dummies(df_lag)
df_lag.shape


# In[18]:


# Define target variable as y
y = df_lag.GDPpcGrowthMA


# In[19]:


df_lag.drop(["GDPpcGrowthMA"], axis=1, inplace = True)


# In[20]:


# create training and test test
X_train, X_test, y_train, y_test = train_test_split(df_lag, y, test_size=0.33,random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ##### Scale data

# In[21]:


# Scale Data for better model performance
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# ### 4. Training and evaluating ML-Models

# In[22]:


# import dependencies for ML Models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Define Models
models = [
    RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42),
    LinearSVR(random_state=42),
    ElasticNet(random_state=42),
    AdaBoostRegressor(random_state=42),
    XGBRegressor(random_state=42),
    ExtraTreesRegressor(random_state=42),
    DecisionTreeRegressor(random_state=42),
]
# Apply k-fold Cross Validation
CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
modelsx = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train_s, y_train, scoring='neg_mean_squared_error', cv=CV, )
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
        modelsx.append(model)
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'neg_mse'])

sns.boxplot(x='model_name', y='neg_mse', data=cv_df)
sns.stripplot(x='model_name', y='neg_mse', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# In[23]:


# output results in mse
print("MSE for models:")
print((-1)*cv_df.groupby('model_name').neg_mse.mean())


# In[24]:


# output results in rmse showing the error in percentage points
print("RMSE for models:")
print(np.sqrt((-1)*(cv_df.groupby('model_name').neg_mse.mean())))


# #### Feature importance

# In[25]:


# "The higher, the more important the feature. The importance of a feature is computed 
# as the (normalized)total reduction of the criterion brought by that feature. 
# It is also known as the Gini importance."
# source - sklearn documentation: 
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html


# In[26]:


# Examplary ExtraTreesRegressor top10 features importances
modelsx[50].fit(X_train_s, y_train)

df_impo = pd.DataFrame()
df_impo["Name"] = pd.DataFrame(X_train).columns
df_impo["Importances"] = modelsx[50].feature_importances_
df_impo = df_impo.sort_values(by=['Importances'], ascending=False)
df_impo.head(10)


# In[27]:


# Exemplary DecisionTreeRegressor top10 features importances
modelsx[60].fit(X_train_s, y_train)

df_impo = pd.DataFrame()
df_impo["Name"] = pd.DataFrame(X_train).columns
df_impo["Importances"] = modelsx[60].feature_importances_
df_impo = df_impo.sort_values(by=['Importances'], ascending=False)
df_impo.head(10)


# ##### Exporting DecisionTree Visualization

# from sklearn.tree import export_graphviz
# export_graphviz(modelsx[60],
#                 feature_names = df_lag.columns,
#                 filled=True,
#                 rounded=True,
#                 out_file='DecisionTree_modelsx60_123.dot')
