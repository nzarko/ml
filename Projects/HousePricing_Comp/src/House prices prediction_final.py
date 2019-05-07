#!/usr/bin/env python
# coding: utf-8

# This competition is very important to me as  it helped me to gain alot of knownledge and experience with mutlifeatured regression. I've read a lot of great notebooks but these two has inspired me to write this kernel.
# 
# 1. [Stacked Regressions to predict House Prices][2] 
# 2. [Regularized Linear Models][3] by **Alexandru Papiu**  : Great Starter kernel on modelling and Cross-validation
# 
# The overall approach is  hopefully concise and easy to follow.. 
# 
# It is pretty much :
# 
# - **Imputing missing values**  by proceeding sequentially through the data
# 
# - **Transforming** some numerical variables that seem really categorical
# 
# - **Label Encoding** some categorical variables that may contain information in their ordering set
# 
# -  [**Box Cox Transformation**][4] of skewed features (instead of log-transformation).
# 
# - **Getting dummy variables** for categorical features. 
# - **Modeling** 
# 
# Then we choose many base models (mostly sklearn based models + sklearn API of  DMLC's [XGBoost][5], cross-validate them on the data, find the model with the best score and submit the results.
# 
#   [1]: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#   [2]:https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
#   [3]: https://www.kaggle.com/apapiu/regularized-linear-models
#   [4]: http://onlinestatbook.com/2/transformations/box-cox.html
#   [5]: https://github.com/dmlc/xgboost
#  [6]: https://github.com/Microsoft/LightGBM
#  [7]: https://www.kaggle.com/humananalog/xgboost-lasso

# In[1]:


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

# In[2]:


train_df = pd.read_csv(DATA_DIR+'/train.csv')
test_df = pd.read_csv(DATA_DIR+'/test.csv')


# Get an overview of the data

# In[3]:


train_df.head(5)


# In[4]:


test_df.head(5)


# Get the shape of train_df and test_df before data manipulation

# In[5]:


train_df.shape


# In[6]:


test_df.shape


# Save the Id column

# In[7]:


train_ID = train_df['Id']
test_ID = test_df['Id']


# Delete the Id column since it's not necessary for the prediction.

# In[8]:


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

# In[9]:


fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers.
# Therefore, we can safely delete them.

# In[10]:


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

# In[11]:


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

# In[12]:


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

# In[13]:


ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ### Missing Data

# In[14]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# Visualize the results.

# In[15]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# **Data Correlation**

# In[16]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train_df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# ### Imputing missing values 

# We impute them by proceeding sequentially through features with missing values

# - **PoolQC** : data description says NA means "No  Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general. 

# In[17]:


all_data['PoolQC'] = all_data['PoolQC'].fillna('None')


# - **MiscFeature** : data description says NA means "no misc feature"

# In[18]:


all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')


# - **Alley** : data description says NA means "no alley access"

# In[19]:


all_data['Alley'] = all_data['Alley'].fillna('None')


# - **Fence** : data description says NA means "no fence"

# In[20]:


all_data['Fence'] = all_data['Fence'].fillna('None')


# - **FireplaceQu** : data description says NA means "no fireplace"

# In[21]:


all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')


# - **LotFrontage** : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can **fill in missing values by the median LotFrontage of the neighborhood**.

# In[22]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))


# - **GarageType, GarageFinish, GarageQual and GarageCond** : Fill missing data with None

# In[23]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')


# - **GarageYrBlt, GarageArea and GarageCars** : Replacing missing data with 0 (Since No garage = no cars in such garage.)

# In[24]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# - **BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath** : missing values are likely zero for having no basement

# In[25]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# - **BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2** : For all these categorical basement-related features, NaN means that there is no  basement.

# In[26]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')


# - **MasVnrArea and MasVnrType** : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type. 

# In[27]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# - **MSZoning (The general zoning classification)** :  'RL' is by far  the most common value.  So we can fill in missing values with 'RL'

# In[28]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# - **Utilities** : For this categorical feature all records are "AllPub", except for one "NoSeWa"  and 2 NA . Since the house with 'NoSewa' is in the training set, **this feature won't help in predictive modelling**. We can then safely  remove it.

# In[29]:


all_data = all_data.drop(['Utilities'], axis=1)


# - **Functional** : data description says NA means typical

# In[30]:


all_data['Functional'] = all_data['Functional'].fillna('Typ')


# - **Electrical** : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

# In[31]:


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


# - **KitchenQual**: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent)  for the missing value in KitchenQual.

# In[32]:


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# - **Exterior1st and Exterior2nd** : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string

# In[33]:


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


# - **SaleType** : Fill in again with most frequent which is "WD"

# In[34]:


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# - **MSSubClass** : NA most likely means No building class. We can replace missing values with None

# In[35]:


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# Check remaining missing values if any 

# In[36]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# No more missing data.

# ### More features engeneering

# **Transforming some numerical variables that are really categorical**

# In[37]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# **Label Encoding some categorical variables that may contain information in their ordering set** 

# In[38]:


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

# In[39]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# **Skewed features**

# In[40]:


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

# In[41]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)


# **Getting dummy categorical features**

# In[42]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# Getting the new train and test sets. 

# In[43]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# # Modelling

# **Import libraries**

# In[44]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, LassoCV, LassoLarsCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


# **Define a cross validation strategy**

# In[45]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# ## Base models

# In[46]:


model_ridge = Ridge()


# <p>The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data. </p>
# <p>Also lets make our model more robust to outliers. We can do that using RobustScaler function via pipe.</p>
# 

# In[47]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmsle_cv(make_pipeline(RobustScaler(), Ridge(alpha = alpha, random_state=1))).mean() 
            for alpha in alphas]


# In[48]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmsle")


# Note the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model begins to overfit. A value of alpha = 10 is about right based on the plot above.

# In[49]:


cv_ridge.min()


# In[50]:


model_ridge = make_pipeline(RobustScaler(), Ridge(alpha = 10, random_state=1))
ridge_res = rmsle_cv(model_ridge)
print('Ridge evaluation result : {:<8.3f}'.format(ridge_res.min()))


# Create a Dataframe mith scores

# In[51]:


pd_scores = pd.DataFrame(data={'Model':['Ridge'], 'Mean':[ridge_res.mean()], 'Std':[ridge_res.std()],'Min':[ridge_res.min()]})


# In[52]:


pd_scores


# - **LassoCV Regression**

# Let' try out the Lasso model. We will do a slightly different approach here and use the built in Lasso CV to figure out the best alpha for us. For some reason the alphas in Lasso CV are really the inverse for the alphas in Ridge.
# Lets make it more robust on outliers using the same technique as above

# In[53]:


model_lassocv = make_pipeline(RobustScaler(), LassoCV(alphas = [1, 0.1, 0.001, 0.0005], random_state=1))
lassocv_res = rmsle_cv(model_lassocv)
print('LassoCV Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(lassocv_res.mean(), lassocv_res.min(),lassocv_res.std()))


# <p>LassoCV performs better than ridge.</p>
# <p>Also LassoCV it does feature selection - setting coefficients of features it deems unimportant to zero. Let's analyze the coefficients:</p>

# In[54]:


coef = pd.Series(model_lassocv.steps[1][1].fit(train,y_train).coef_, index = train.columns)


# In[55]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# <p>One thing to note here however is that the features selected are not necessarily the "correct" ones - especially since there are a lot of collinear features in this dataset. One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is.</p>
# <p> We can also take a look directly at what the most important coefficients are: <p>

# In[56]:


imp_coef = pd.concat([coef.sort_values().head(12),
                     coef.sort_values().tail(12)])


# In[57]:


plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# In[58]:


#let's look at the residuals as well:
plt.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lassocv.steps[1][1].predict(train), "true":y_train})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")


# <p>The residual plot looks pretty good.</p>
# <p>Add LassoCV scores to scores dataframe.</p>

# In[59]:


lasso_score=[{'Model':'LassoCV', 'Mean':lassocv_res.mean(),'Std':lassocv_res.std(), 'Min':lassocv_res.min()}]
pd_scores = pd_scores.append(lasso_score,ignore_index=True, sort=False)


# In[60]:


pd_scores


# - **Elastic Net Regression**
# <br>
# again make robust to outliers

# In[61]:


model_enet_cv = make_pipeline(RobustScaler(), ElasticNetCV(cv=None, random_state=0))


# In[62]:


enet_score = rmsle_cv(model_enet_cv)
print('ElasticNet Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(enet_score.mean(), enet_score.min(),enet_score.std()))


# Add score to scores data frame

# In[63]:


score = [{'Model':'ElasticNetCV', 'Mean':enet_score.mean(),'Std':enet_score.std(), 'Min':enet_score.min()}]
pd_scores = pd_scores.append(score,ignore_index=True, sort=False)


# - **Kernel Ridge Regression**

# In[64]:


from sklearn.kernel_ridge import KernelRidge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR_Score = rmsle_cv(KRR)
print('KernelRidge Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(KRR_Score.mean(), KRR_Score.min(),KRR_Score.std()))


# In[65]:


score = [{'Model':'KernelRidge', 'Mean':KRR_Score.mean(),'Std':KRR_Score.std(), 'Min':KRR_Score.min()}]
pd_scores = pd_scores.append(score,ignore_index=True, sort=False)


# - **xgboost**

# Adding an xgboost model to see if we can improve our score

# In[66]:


import xgboost as xgb


# In[67]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score1 = rmsle_cv(model_xgb)
print('xgboost Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(score1.mean(), score1.min(),score1.std()))


# In[68]:


# dtrain = xgb.DMatrix(train, label = y_train)
# dtest = xgb.DMatrix(test)

# params = {"max_depth":2, "eta":0.1}
# model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

# model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

# model_xgb1 = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
# score1 = rmsle_cv(model_xgb1)
# print('xgboost Score : {:<8.4f}, with min value : {:<8.4f} and std : {:<8.4f}'.format(score1.mean(), score1.min(),score1.std()))
#model_xgb.fit(train, y_train)

# Adding xgboost score to scores board
score = [{'Model':'XGBoost', 'Mean':score1.mean(),'Std':score1.std(), 'Min':score1.min()}]
pd_scores = pd_scores.append(score,ignore_index=True, sort=False)


# In[69]:


pd_scores.sort_values(by=['Mean','Std', 'Min'])


# LassoCV gave us the best result. Lets fit the model and submit the results.

# In[70]:


lassocv = model_lassocv.steps[1][1]
lassocv.fit(train,y_train)
preds = np.expm1(lassocv.predict(test))
preds


# In[71]:


solution = pd.DataFrame({"id":test_ID, "SalePrice":preds})
solution.head(5)


# In[72]:


solution.to_csv("lasso_sol.csv", index = False)


# In[ ]:




