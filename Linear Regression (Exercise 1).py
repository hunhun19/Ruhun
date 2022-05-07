#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston


# In[2]:


boston = load_boston()

print(boston.DESCR)


# In[3]:


plt.hist(boston.target, bins=50)


# In[4]:


plt.scatter(boston.data[:,5],boston.target)

plt.ylabel('Price in $1000s')
plt.xlabel('Number of rooms')


# In[5]:


boston_df = pd.DataFrame(boston.data)

boston_df.columns = boston.feature_names

boston_df.head()


# In[6]:


boston_df['Price'] = boston.target

boston_df.head()


# In[7]:


sns.lmplot(x='RM', y='Price', data=boston_df)


# In[8]:


#Set independent variabel

X = np.array([boston_df.RM], dtype='float')

#Set dependent variabel

Y = np.array([boston_df.Price], dtype='float').T


# In[9]:


A = np.vstack([X, np.ones(506)]).T


# In[10]:


#Return the least-squares solution to a linear matrix equation.

m ,b = np.linalg.lstsq(A, Y, rcond=None)[0]


# In[11]:


#Price vs Avg Number of Rooms
plt.plot(boston_df.RM, boston_df.Price, 'o')

x=boston_df.RM
plt.plot(x, m*x + b, 'r', label='Best Fit Line')


# In[12]:


#RMSE = measure of how spread out these residuals are
#error value = sum of square residual value

result = np.linalg.lstsq(A,Y, rcond=None)

error_value = result[1]

rsme = np.sqrt(error_value/len(A))

print('Root Mean Square Error = %.2f' %rsme)


# In[13]:


#R^2 = indicates the percentage of the variance in the dependent variable that the independent variables explain collectively


# In[14]:


import sklearn
from sklearn.linear_model import LinearRegression

lreg = LinearRegression()


# In[15]:


#Set independent and dependent variabel for model

X_multi = boston_df.drop('Price', axis=1)

Y_target = boston_df['Price']


# In[16]:


lreg.fit(X_multi, Y_target)

print('Intercept of model = %.2f' %lreg.intercept_)
print('Number of Coefficient = %.2f' %len(lreg.coef_))
print(lreg.coef_)


# In[17]:


#Assign the coefficient into a dataframe

coef_df = pd.DataFrame(boston_df.columns)

coef_df.columns = ['Features']

coef_df['Coefficient'] =pd.Series(lreg.coef_)

coef_df.drop([13], axis=0, inplace=True)

coef_df


# In[38]:


#Split the data
K = np.vstack(boston_df.RM)

K_train, K_test, Y_train, Y_test = sklearn.model_selection.train_test_split(K, boston_df.Price)

#Fit the data
lreg = LinearRegression()

lreg.fit(K_train, Y_train)

print(lreg.coef_)
print(lreg.intercept_)


# In[42]:


#Predict the data
predict_train = lreg.predict(K_train)
predict_test = lreg.predict(K_test)

#The Score of Prediction
print(lreg.score(K_train, Y_train))
print(lreg.score(K_test, Y_test))


# In[46]:


#RSME
rsme_train = np.sqrt(((Y_train - predict_train)**2).mean())
rsme_test = np.sqrt(((Y_test - predict_test)**2).mean())

print(rsme_train)
print(rsme_test)


# In[56]:


#Residual Plots

train =plt.scatter(predict_train, (Y_train - predict_train), c='b', alpha=0.5)

test = plt.scatter(predict_test, (Y_test - predict_test), c='r', alpha=0.5)

plt.hlines(y=0, xmin=-10, xmax=50)

plt.legend(['Training', 'Test'], loc='lower right')

plt.title('Residual_Plot')

