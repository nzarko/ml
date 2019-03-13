
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