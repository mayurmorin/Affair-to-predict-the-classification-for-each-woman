
# coding: utf-8

# <h4>Extramarital Affairs Dataset</h4>
# 
# The dataset is affairs dataset and comes with Statsmodels. It was derived from a survey of women in 1974 by Redbook magazine, in which married women were asked about their participation in extramarital affairs. More information about the study is available in a 1978 paper from the Journal of Political Economy.
# 
# Description of Variables
# The dataset contains 6366 observations of 9 variables:
# 
# <li>rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good) <br>
# <li>age: woman's age <br>
# <li>yrs_married: number of years married <br>
# <li>children: number of children <br>
# <li>religious: woman's rating of how religious she is (1 = not religious, 4 = strongly religious) <br>
# <li>educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 = college  <br> graduate, 17 = some graduate school, 20 = advanced degree) <br>
# <li>occupation: woman's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 = "white collar", <br> 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 = professional with advanced degree) <br>
# <li>occupation_husb: husband's occupation (same coding as above) <br>
# <li>affairs: time spent in extra-marital affairs <br>

# <h4>Problem Statement</h4>
# We treat this as a classification problem by creating a new binary variable affair (did the woman have at least one affair?) and try to predict the classification for each woman.

# <h4>Import Modules</h4>

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# <h4>Preprocessing</h4>

# In[2]:


# load dataset
dta = sm.datasets.fair.load_pandas().data

# add "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)


# <h4>Data Exploration</h4>

# In[3]:


dta.groupby('affair').mean()


# We can see that on average, women who have affairs rate their marriages lower, 
# which is to be expected. Let's take another look at the rate_marriage variable.

# In[4]:


dta.groupby('rate_marriage').mean()


# An increase in age, yrs_married, and children appears to correlate with a declining marriage rating.

# <h4>Data Visualization</h4>

# In[5]:


# show plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# histogram of education
dta.educ.hist()
plt.title('Histogram of Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')


# In[6]:


# histogram of marriage rating
dta.rate_marriage.hist()
plt.title('Histogram of Marriage Rating')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')


# Let's take a look at the distribution of marriage ratings for those having affairs versus those not having affairs.

# In[7]:


# barplot of marriage rating grouped by affair (True or False)
pd.crosstab(dta.rate_marriage, dta.affair.astype(bool)).plot(kind='bar')
plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')


# Let's use a stacked barplot to look at the percentage of women having affairs by number of years of marriage.

# In[8]:


affair_yrs_married = pd.crosstab(dta.yrs_married, dta.affair.astype(bool))
affair_yrs_married.div(affair_yrs_married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Affair Percentage by Years Married')
plt.xlabel('Years Married')
plt.ylabel('Percentage')


# <h4>Prepare Data for Logistic Regression</h4>

# In[9]:


# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children +                   religious + educ + C(occupation) + C(occupation_husb)',
                  dta, return_type="dataframe")
# print X.columns

# rename column names for the dummy variables for better looks:
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})

# and flatten y into a 1-D array so that scikit-learn will properly understand it as the response variable.
y = np.ravel(y)


# <h4>Logistic Regression</h4>

# In[10]:


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)


# 72% accuracy may seem OK, but what's the accuracy if we simply predict no for all observations in the dataset?

# In[11]:


# what percentage had affairs?
y.mean()


# Only 32% of the women had affairs, which means that you could obtain 68% accuracy by always predicting no. So we're doing better than the null error rate, but not by much.

# Increases in marriage rating and religiousness correspond to a decrease in the likelihood of having an affair. For both the wife's occupation and the husband's occupation, the lowest likelihood of having an affair corresponds to the baseline occupation (student), since all of the dummy coefficients are positive.

# In[14]:


# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# predict class labels for the test set
predicted = model2.predict(X_test)
predicted

# generate class probabilities
probs = model2.predict_proba(X_test)
probs


# The accuracy is 73%, which is the same as we experienced when training and predicting on the same data.

# In[16]:


# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))


# In[17]:


print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))


# <h4>Model Evaluation Using Cross-Validation</h4>

# In[18]:


# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
scores, scores.mean()


# Still performing at 73% accuracy.

# <h4>Predicting the Probability of an Affair</h4>

# Let's predict the probability of an affair for a random woman not present in the dataset. She's a 25-year-old teacher who graduated college, has been married for 3 years, has 1 child, rates herself as strongly religious, rates her marriage as fair, and her husband is a farmer.

# In[19]:


model.predict_proba(np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 3, 25, 3, 1, 4,
                              16]]))


# The predicted probability of an affair is 23%.

# <h4>Improving the Model</h4>
# The following could be tried to improve the model:
# 
# <li>including interaction terms
# <li>removing features
# <li>regularization techniques
# <li>using a non-linear model
