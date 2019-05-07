#!/usr/bin/env python
# coding: utf-8

# Enter Title

# In[2]:


import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


import os
DATA_DIR='../input'
print(os.listdir(DATA_DIR))


# Load train and test datasets

# In[3]:


train_df = pd.read_csv(DATA_DIR+'/train.csv')
test_df = pd.read_csv(DATA_DIR+'/test.csv')


# Get an overview of the data

# In[4]:


train_df.head(5)


# In[5]:


test_df.head(5)


# Get the shape of train_df and test_df before data manipulation

# In[6]:


train_df.shape


# In[7]:


test_df.shape


# Save the Id column

# In[8]:


train_ID = train_df['Id']
test_ID = test_df['Id']


# Delete the Id column since it's not necessary for the prediction.

# In[10]:


#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train_df.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test_df.shape))


# <h1>Data Processing</h1>

# ## Outliers

# [Documentation](http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt)
# for the Ames Housing Data indicates that there are outliers present in the training data

# Lets explore these outliers

# In[13]:


fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers.
# Therefore, we can safely delete them.

# In[14]:


#Deleting outliers
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# ### Note : 
# Outliers removal is note always safe.  We decided to delete these two as they 
# are very huge and  really  bad ( extremely large areas for very low  prices). 
# There are probably others outliers in the training data.   
# However, removing all them  may affect badly our models if ever there were also
# outliers  in the test data. That's why , instead of removing them all, we will 
# just manage to make some of our  models robust on them. You can refer to  the 
# modelling part of this notebook for that. 

# ## Target Variable

# **SalePrice** is the variable we need to predict. So let's do some analysis on this variable first.

# In[16]:


sns.distplot(train_df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()


# The target variable is right skewed. We need to transform this variable and make it more normally distributed.
# One way to do it is to apply log(1+x) to all elements of the column using the numpy function log1p.

# In[17]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

#Check the new distribution 
sns.distplot(train_df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()


# The skew seems now corrected and the data appears more normally distributed. 

# ## Feature engineering

# let's first concatenate the train and test data in the same dataframe

# In[19]:


ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ### Missing Data

# In[20]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# Visualize the results.

# In[21]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# **Data Correlation**

# In[23]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train_df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# ### Imputing missing values 

# We impute them by proceeding sequentially through features with missing values

# - **PoolQC** : data description says NA means "No  Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general. 

# In[25]:


all_data['PoolQC'] = all_data['PoolQC'].fillna('None')


# - **MiscFeature** : data description says NA means "no misc feature"

# In[27]:


all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')


# - **Alley** : data description says NA means "no alley access"

# In[29]:


all_data['Alley'] = all_data['Alley'].fillna('None')


# - **Fence** : data description says NA means "no fence"

# In[30]:


all_data['Fence'] = all_data['Fence'].fillna('None')


# - **FireplaceQu** : data description says NA means "no fireplace"

# In[31]:


all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')


# - **LotFrontage** : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can **fill in missing values by the median LotFrontage of the neighborhood**.

# In[33]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))


# - **GarageType, GarageFinish, GarageQual and GarageCond** : Fill missing data with None

# In[34]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')


# - **GarageYrBlt, GarageArea and GarageCars** : Replacing missing data with 0 (Since No garage = no cars in such garage.)

# In[35]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# - **BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath** : missing values are likely zero for having no basement

# In[36]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# - **BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2** : For all these categorical basement-related features, NaN means that there is no  basement.

# In[37]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')


# - **MasVnrArea and MasVnrType** : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type. 

# In[39]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# - **MSZoning (The general zoning classification)** :  'RL' is by far  the most common value.  So we can fill in missing values with 'RL'

# In[40]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# - **Utilities** : For this categorical feature all records are "AllPub", except for one "NoSeWa"  and 2 NA . Since the house with 'NoSewa' is in the training set, **this feature won't help in predictive modelling**. We can then safely  remove it.

# In[41]:


all_data = all_data.drop(['Utilities'], axis=1)


# - **Functional** : data description says NA means typical

# In[42]:


all_data['Functional'] = all_data['Functional'].fillna('Typ')


# - **Electrical** : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

# In[43]:


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


# - **KitchenQual**: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent)  for the missing value in KitchenQual.

# In[44]:


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# - **Exterior1st and Exterior2nd** : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string

# In[45]:


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


# - **SaleType** : Fill in again with most frequent which is "WD"

# In[46]:


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# - **MSSubClass** : NA most likely means No building class. We can replace missing values with None

# In[47]:


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# Check remaining missing values if any 

# In[48]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# No more missing data.

# ### More features engeneering

# **Transforming some numerical variables that are really categorical**

# In[50]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# **Label Encoding some categorical variables that may contain information in their ordering set** 

# In[51]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# **Adding one more important feature**

# Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house

# In[52]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# **Skewed features**

# In[53]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# **Box Cox Transformation of (highly) skewed features**

# We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+𝑥 .
# 
# Note that setting  𝜆=0  is equivalent to log1p used above for the target variable.
# 
# See [this page](http://onlinestatbook.com/2/transformations/box-cox.html) for more details on Box Cox Transformation as well as [the scipy function's page](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.special.boxcox1p.html)

# In[54]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)


# **Getting dummy categorical features**

# In[55]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# Getting the new train and test sets. 

# In[56]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# # Modelling

# **Import libraries**

# In[69]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


# **Define a cross validation strategy**

# In[60]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# ## Base models

# In[61]:


model_ridge = Ridge()


# The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.

# In[63]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmsle_cv(make_pipeline(RobustScaler(), Ridge(alpha = alpha, random_state=1))).mean() 
            for alpha in alphas]


# In[64]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmsle")


# Note the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model begins to overfit. A value of alpha = 10 is about right based on the plot above.

# In[65]:


# cv_ridge.min()
cv_ridge

# Lets do the procedure again, but now lets make it more robust to outliers. For that we use the sklearn's Robustscaler() method on pipeline

# In[71]:


model_ridge = make_pipeline(RobustScaler(), Ridge(alpha = 10, random_state=1))
ridge_res = rmsle_cv(model_ridge)


# In[72]:


# ridge_res


# In[73]:


ridge_res.min()


# In[ ]:




