#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
os.getcwd()
os.chdir('/Users/utente/Downloads')


# ## Question 1 

# In[13]:


import pandas as pd
#Pandas version (1 point)

pd.__version__


# ## Question 2

# In[96]:


#Records count (1 point)
df = pd.read_csv('ML DATA.txt')
df.shape


# ## Question 3

# In[15]:


#Laptop brands (1 point)
df['Brand'].nunique()


# ## Question 4

# In[16]:


#How many columns in the dataset have missing values?
df.isnull().sum()


# ## Question 5.

# In[52]:


#Maximum final price (1 point)
#What's the maximum final price of Dell notebooks in the dataset?

DELL = df[df['Brand'] == 'Dell']

DELL['Final Price'].max()


# ## Question 6

# In[ ]:


'''  Median value of Screen

Find the median value of Screen column in the dataset.
Next, calculate the most frequent value of the same Screen column.
Use fillna method to fill the missing values in Screen column with the most frequent value from the previous step.
Now, calculate the median value of Screen once again.
Has it changed?'''



# In[39]:


df['Screen'].median()
df['Screen'].mode()


# In[105]:


df_filled = df['Screen'].fillna(15.6)
df['Screen'].fillna(15.6)


# In[106]:


df['Screen'].median()


# ## Question 7

# In[ ]:


''' Select all the "Innjoo" laptops from the dataset.
Select only columns RAM, Storage, Screen.
Get the underlying NumPy array. Let's call it X.
Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
Compute the inverse of XTX.
Create an array y with values [1100, 1300, 800, 900, 1000, 1100].
Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
What's the sum of all the elements of the result?'''


# In[69]:


X = df[df['Brand'] == 'Innjoo']
    
X =X[['RAM','Storage', 'Screen']]
X


# In[70]:


Trans = X.T
Trans 


# In[76]:


Matrix_Multiplication = Trans.dot(X)
Matrix_Multiplication


# In[81]:


import numpy as np
inverse = np.linalg.inv(Matrix_Multiplication)
inverse 


# In[80]:


y = np.array([1100, 1300, 800, 900, 1000, 1100])
y


# In[88]:


INVTRANS = (inverse).dot(Trans)


# In[90]:


w = INVTRANS.dot(y)
w


# In[91]:


w.sum()


# In[ ]:




