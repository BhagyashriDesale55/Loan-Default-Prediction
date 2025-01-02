#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loans = pd.read_csv('loan_data.csv')


# In[3]:


loans.info()


# In[4]:


loans.describe()


# In[5]:


loans.head()


# In[6]:


#eda
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[7]:


plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# In[8]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# In[9]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# In[10]:


#Setting up the data
loans.info()


# In[11]:


cat_feats = ['purpose']


# In[12]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[13]:


final_data.info()


# In[14]:


#train test split
from sklearn.model_selection import train_test_split


# In[15]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[16]:


#training the decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[17]:


#prediction and evaluation of decision tree
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[19]:


print(confusion_matrix(y_test,predictions))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualize the decision tree
plt.figure(figsize=(20,10))  # Set the figure size
plot_tree(dtree, feature_names=X.columns, class_names=['Fully Paid', 'Not Fully Paid'], filled=True, rounded=True, fontsize=10)
plt.show()


print("For a single node")


# Visualizing a single decision tree
plt.figure(figsize=(12,8))
plot_tree(dtree, feature_names=X_train.columns, class_names=['Fully Paid', 'Not Fully Paid'], max_depth=1, filled=True)
plt.show()


# In[21]:


#training the random forest 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)


# In[22]:


#prediction and evaluation
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[23]:


#confusion matrix
print(confusion_matrix(y_test,predictions))


# In[ ]:




