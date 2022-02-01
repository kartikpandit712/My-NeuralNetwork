#!/usr/bin/env python
# coding: utf-8

# # Keras: Deep learning library mianly used for implementing Neural Network
# 
# Features: 
# Its support Linear Algorithms
# Its supports polynomials
# Its also supports classification
# 
# # How to Process;
# 1.loading dataset
# 2.define model(layers)
# 3.compile model
# 4.fitting model
# 5.Evalauting model
# 
# # Dense: is a class
# Number of Neurons:
# Input dimen input_dim
# activation function 'relu','sigmoid','softmax'
# 
# #intermidate layer - relu activation function
# #output layer- sigmoid activation function

# # Nureal Network with Classification

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np


# In[ ]:


ds=np.loadtxt(r"C:\Users\karti\Videos\Python\diabetes.csv",delimiter=",")
#print(ds)
X=ds[:,0:8]
Y=ds[:,8]


# In[ ]:


model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.fit(X,Y,epochs=500,batch_size=4)


# In[ ]:


scores=model.evaluate(X,Y)
acc=scores[1]*100
print(acc)


# In[ ]:


y=model.predict([[6,148,72,35,0,33.6,0.627,50]])
print(y)


# In[ ]:




