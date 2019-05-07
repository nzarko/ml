# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Utilities for data proccesing
def dropna_p(df, min_percent=0.7):
        out_df = df[[column for column in df if df[column].isna().mean() < min_percent]]
        print('List of dropped coluns : ', end=' ')
        for c in df.columns:
            if c not in out_df.columns:
                print(c, end=', ')
            
        return out_df
    
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
    

def show_features_importances(forest,columns, nfirst=X.shape[1]):
    # forest.fit(X, y)
    result = []
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(nfirst):
        print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))
        result.append(columns[indices[f]])
    
    # Plot the feature importances of the forest
#    plt.figure()
#    plt.title("Feature importances")
#    plt.bar(range(X.shape[1]), importances[indices],
#           color="r", yerr=std[indices], align="center")
#    plt.xticks(range(X.shape[1]), indices)
#    plt.xlim([-1, X.shape[1]])
#    plt.show()
    
    return result


def draw_tree(regressor, cn):
    # Extract single tree
    estimator = regressor.estimators_[5]
    
    from sklearn.tree import export_graphviz
    
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot', 
                    feature_names = train_df.columns[:-1],
                    class_names = cn,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    
    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    
    # Display in jupyter notebook
    from IPython.display import Image
    Image(filename = 'tree.png')
#
#new_train_df= dropna_p(train_df)
## Get rid of Id
#train_id_col = new_train_df['Id']
#new_train_df = new_train_df.drop(['Id'],axis=1);


# Importing the dataset
dataset = pd.read_csv('train.csv')

datasetID = dataset['Id']
dataset = dataset.drop(['Id'],axis=1)

# Drop columns with more than 75% of NaNs
train_df = dropna_p(dataset)
dataset.shape

# Get dummies
train_df = pd.get_dummies(train_df)
train_df = train_df.fillna(train_df.mean())

# Rearrange data so the SalePrice be the last column
SalePrice = train_df['SalePrice']
train_df = train_df.drop(['SalePrice'], axis = 1)
train_df['SalePrice'] = SalePrice

X = train_df.iloc[:,:-1].values
y = train_df.iloc[:,-1].values
# y = train_df['SalePrice'].iloc[:].values
train_df.columns.get_loc('SalePrice')

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 250, random_state = 0)
regressor.fit(X, y)

# Show and plot features importances
col_names_rf = show_features_importances(regressor, train_df.columns,12)



# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pandas.read_csv(url, names=names)
# array = dataframe.values
#X = array[:,0:8]
# Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=12)
fit = test.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])
indices = np.argsort(features)[::-1]
indices
    
# Print the feature ranking
print("Feature ranking:")   
print("feature %s " % (train_df.columns[indices[1]]))


from sklearn.feature_selection import RFE

# feature extraction

rfe = RFE(regressor, 12)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
col_names=[]
for i in range(len(fit.ranking_)):
    if fit.ranking_[i] == 1:
        col_names.append(train_df.columns[i])
        
print ('Selected features names : %s ' % col_names)



from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

model_ridge = Ridge()

#The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

cv_ridge.min()