#!/usr/bin/env python
# coding: utf-8

# # WEEK - 2 ASSIGNMENTS
# ## Data Science & Data Analytics Laboratory
# ### Name: Kinjal Raykarmakar
# #### Section: CSE 3H
# #### Roll No.: 29
# #### Enrollment No.: 12018009019439

# # Regression - Diabetes

# In[2]:


from sklearn.datasets import load_diabetes


# In[3]:


diabetes = load_diabetes()


# In[4]:


print(diabetes.DESCR) # DESCR stands for description


# In[5]:


import pandas as pd


# In[7]:


data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)


# In[9]:


data['y'] = pd.DataFrame(diabetes.target)


# In[10]:


data.head()


# In[13]:


corr_mat = pd.DataFrame(data.corr().round(2))
corr_mat


# In[16]:


max_pos_corr = corr_mat["y"][:-1].idxmax()
max_pos_corr


# In[17]:


x = data[max_pos_corr]
y = data['y']


# In[18]:


pd.DataFrame([x, y]).transpose().head()


# In[19]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[20]:


x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)


# In[21]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)


# In[23]:


y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

print(sqrt(mean_squared_error(y_pred, y_test)))


# In[24]:


print("y = {} {} + {}".format(model.coef_[0].round(2),  max_pos_corr, model.intercept_.round(2)))


# In[25]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


plt.scatter(x_test, y_test, label="Actual")
plt.plot(x_test, y_pred, color="red", label="Fit")
plt.xlabel(max_pos_corr)
plt.ylabel("y")
plt.legend()
plt.show()

