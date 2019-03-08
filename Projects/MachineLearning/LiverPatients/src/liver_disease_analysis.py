
# coding: utf-8

#  Liver Desease Analysis

# In[39]:


#import the neccessary modules
# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')
#Import all required libraries for reading data, analysing and visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder


# In[5]:


#Read the training & test data
dataset = pd.read_csv('../input/indian_liver_patient.csv')


# In[6]:


dataset.head()


# In[7]:


# View dataset info
dataset.info()


# In[9]:


#Describe gives statistical information about NUMERICAL columns in the dataset
dataset.describe(include='all')
#We can see that there are missing values for Albumin_and_Globulin_Ratio as only 579 entries have valid values indicating 4 missing values.
#Gender has only 2 values - Male/Female


# In[11]:


dataset.columns


# In[13]:


dataset['Dataset'][:20]


# In[15]:


#Check for any null values
dataset.isnull().sum()


# <p> Check if Albumin_and_Globulin_Ratio is important feature </p
#     <h1 id="Data-Visualization">Data Visualization<a class="anchor-link" href="#Data-Visualization" target="_self">¶</a></h1>

# In[16]:


sns.countplot(data=dataset, x='Dataset', label='Count')
LD, NLD = dataset['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)


# In[18]:


sns.countplot(data=dataset, x = 'Gender', label='Count')

M, F = dataset['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)


# In[28]:


sns.catplot(x="Age", y="Gender", hue="Dataset",kind='point', data=dataset);


# In[29]:


dataset[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).count().sort_values(by='Dataset', ascending=False)


# In[30]:


dataset[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=False)


# In[31]:


g = sns.FacetGrid(dataset, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[32]:


g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)


# There seems to be direct relationship between Total_Bilirubin and Direct_Bilirubin. We have the possibility of removing one of this feature.

# In[40]:


sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=dataset, kind="reg")


# In[41]:


sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=dataset, kind="reg")


# In[42]:


g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[43]:


sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=dataset, kind="reg")


# <p>No linear correlation between Alkaline_Phosphotase and Alamine_Aminotransferase</p>

# In[44]:


g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# <p>There is linear relationship between Total_Protiens and Albumin and the gender. We have the possibility of removing one of this feature.</p>

# In[45]:


sns.jointplot("Total_Protiens", "Albumin", data=dataset, kind="reg")


# In[46]:


g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# <p>There is linear relationship between Albumin_and_Globulin_Ratio and Albumin. We have the possibility of removing one of this feature.</p>

# In[47]:


sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=dataset, kind="reg")


# In[48]:


g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# <h1>Observations</h1>
# <div class="">
#         
# <div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
# </div>
# <div class="inner_cell">
# <div class="text_cell_render border-box-sizing rendered_html">
# <p>From the above jointplots and scatterplots, we find direct relationship between the following features:<br>
# Direct_Bilirubin &amp; Total_Bilirubin<br>
# Aspartate_Aminotransferase &amp; Alamine_Aminotransferase<br>
# Total_Protiens &amp; Albumin<br>
# Albumin_and_Globulin_Ratio &amp; Albumin</p>
# <p>Hence, we can very well find that we can omit one of the features. I'm going to keep the follwing features:<br>
# Total_Bilirubin<br>
# Alamine_Aminotransferase<br>
# Total_Protiens<br>
# Albumin_and_Globulin_Ratio<br>
# Albumin</p>
# 
# </div>
# </div>
# </div>

# In[49]:


dataset.head(5)


# <p>Convert Categorical variable Gender to indicator variable</p>

# In[50]:


pd.get_dummies(dataset['Gender'], prefix = 'Gender').head()


# In[51]:


dataset = pd.concat([dataset,pd.get_dummies(dataset['Gender'], prefix = 'Gender')], axis=1)


# In[52]:


dataset.head()


# In[53]:


dataset.describe()


# In[54]:


dataset[dataset['Albumin_and_Globulin_Ratio'].isnull()]


# In[55]:


dataset["Albumin_and_Globulin_Ratio"] = dataset.Albumin_and_Globulin_Ratio.fillna(dataset['Albumin_and_Globulin_Ratio'].mean())


# In[56]:


dataset.head()


# In[57]:


X = dataset.drop(['Gender','Dataset'], axis=1)
X.head(3)


# In[58]:


y = dataset['Dataset'] # 1 for liver disease; 2 for no liver disease


# In[59]:


# Correlation
liver_corr = X.corr()


# In[60]:


liver_corr


# In[61]:


plt.figure(figsize=(30, 30))
sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')
plt.title('Correlation between features');


# <p>#The above correlation also indicates the following correlation<br>
# # Total_Protiens & Albumin<br>
# # Alamine_Aminotransferase & Aspartate_Aminotransferase<br>
# # Direct_Bilirubin & Total_Bilirubin<br>
# # There is some correlation between Albumin_and_Globulin_Ratio and Albumin. <br>
#     But its not as high as Total_Protiens & Albumin</p

# <h1> Machine Learning </h1>

# <p> Import the modules </p>

# In[62]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# <h2> Logistic Regression </h2>

# In[64]:


#2) Logistic Regression
# Create logistic regression object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))
print('Classification Report: \n', classification_report(y_test,log_predicted))

sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")


# In[65]:


coeff_df = pd.DataFrame(X.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# <h2> Gaussian Naive Bayes </h2>

# In[66]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
#Predict Output
gauss_predicted = gaussian.predict(X_test)

gauss_score = round(gaussian.score(X_train, y_train) * 100, 2)
gauss_score_test = round(gaussian.score(X_test, y_test) * 100, 2)
print('Gaussian Score: \n', gauss_score)
print('Gaussian Test Score: \n', gauss_score_test)
print('Accuracy: \n', accuracy_score(y_test, gauss_predicted))
print(confusion_matrix(y_test,gauss_predicted))
print(classification_report(y_test,gauss_predicted))

sns.heatmap(confusion_matrix(y_test,gauss_predicted),annot=True,fmt="d")


# <h2> Random Forest </h2>

# In[67]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#Predict Output
rf_predicted = random_forest.predict(X_test)

random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)
print('Random Forest Score: \n', random_forest_score)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test,rf_predicted))
print(confusion_matrix(y_test,rf_predicted))
print(classification_report(y_test,rf_predicted))


# <h2> Model Evaluation </h2>

# In[68]:


#We can now rank our evaluation of all the models to choose the best one for our problem. 
models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 'Gaussian Naive Bayes','Random Forest'],
    'Score': [ logreg_score, gauss_score, random_forest_score],
    'Test Score': [ logreg_score_test, gauss_score_test, random_forest_score_test]})
models.sort_values(by='Test Score', ascending=False)


# <h3> Linear Regression </h3>

# In[69]:


linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
#Predict Output
lin_predicted = linear.predict(X_test)

linear_score = round(linear.score(X_train, y_train) * 100, 2)
linear_score_test = round(linear.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Linear Regression Score: \n', linear_score)
print('Linear Regression Test Score: \n', linear_score_test)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

from sklearn.feature_selection import RFE
rfe =RFE(linear, n_features_to_select=3)
rfe.fit(X,y)


# In[70]:


for i in range(len(rfe.ranking_)):
    if rfe.ranking_[i] == 1:
        print(X.columns.values[i])


# In[72]:


#I'm considering seven important features based on recursive feature elimination
#finX = liver_df[['Age','Direct_Bilirubin','Total_Protiens','Albumin', 'Gender_Female', 'Gender_Male']]
finX = dataset[['Total_Protiens','Albumin', 'Gender_Male']]
finX.head(4)


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(finX, y, test_size=0.30, random_state=101)


# In[74]:


#Logistic Regression
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))
print('Classification Report: \n', classification_report(y_test,log_predicted))

sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")

