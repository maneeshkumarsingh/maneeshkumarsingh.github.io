#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
import seaborn as sns


# In[2]:


df=pd.read_csv('Churn_Modelling.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df=df.drop('Surname',axis=1)


# In[6]:


hu=pd.get_dummies(df.Gender,drop_first=True)
df=pd.concat([df,hu],axis=1)
df.drop('Gender',axis=1,inplace=True)
hu=pd.get_dummies(df.Geography,drop_first=True)
df=pd.concat([df,hu],axis=1)
df.drop('Geography',axis=1,inplace=True)
df.head()


# In[7]:


x=df.drop('Exited',axis=1)
y=df['Exited']


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[9]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[10]:


X_test.shape


# In[11]:


import keras


# In[12]:


# 2 way difining deel leang layer 
# 1 seq for ann
from keras.models import Sequential
#2 for initi for layer in ann
from keras.layers import Dense


# In[13]:


# nn build  clasifiere 
# inisilis ann for classifire
classifier = Sequential()


# In[14]:


# Adding layere ,, input layer and 1 hidden layer, already know input layer from data , all independ variyebal
classifier.add(Dense(output_dim=6,init='uniform', activation='relu',input_dim=13))


# In[15]:


# second hidden layer
classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))


# In[16]:


# out put lyaer
# dim =1 bcz only one catogory, more then 2 cotg use softmax space of sygmoic
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# In[17]:


# init compli for optimiz wait 
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])


# In[18]:


# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# In[19]:


y_pred=classifier.predict(X_test)


# In[20]:


y_pred=(y_pred > 0.5)


# In[21]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[22]:


cm


# In[23]:


(1543+149)/2000


# In[24]:


x.head()


# In[25]:


new_pred=classifier.predict(sc.transform(np.array([[0,0,600,40,3,60000,2,1,1,5000,1,0,0]])))


# In[26]:


new_pred > 0.5


# In[ ]:


# classifier 


# In[27]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[28]:


def build_classifire():
    classifier = Sequential() #local classifier
    classifier.add(Dense(output_dim=6,init='uniform', activation='relu',input_dim=13))
    classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier


# In[29]:


classifier = KerasClassifier(build_fn=build_classifire,batch_size=10, nb_epoch=100)


# In[ ]:




