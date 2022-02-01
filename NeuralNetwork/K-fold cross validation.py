#!/usr/bin/env python
# coding: utf-8

# its is most popular strategy and it used by data scientist.
# 
# -it is data partitioning strategy.
# -it is mainly based generalization of data so that the splitting of data is proper also the model should work on unseen data.
# 
# -It automatically generalizes data splitting method so that the trained model is accurate in terms of all kind of data.
# 
# # Different approaches:
# 1.using K fold cross validation we can evaluate model's performance.
# 2.we can tune hyperparameter.
# # Keras Regressor:
# it is used to perform regression on large amount of data using neural network.
# 
# step:1.define estimator
# 2.validate estimator
# 3.create pipeline to fit and predict the model data

# In[ ]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[ ]:


df=pd.read_csv(r"C:\Users\karti\Videos\Python\housing.csv",delim_whitespace=True,header=None)
ds=df.values
print(ds)


# In[ ]:


X=ds[:,0:13]    #[row,column]
Y=ds[:,13]


# In[ ]:


#create s basic neural model
def mymodel():
    
    model=Sequential()
    model.add(Dense(13,input_dim=13,kernel_initializer='normal',activation='relu')) #initializer if you do not have the GPU
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model


# In[ ]:


estimators=[]
estimators.append(("standardize",StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=mymodel,epochs=50,batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10)
result=cross_val_score(pipeline,X,Y,cv=kfold)
print(result.mean())


# In[ ]:


pipeline.fit(X,Y)


# In[ ]:


ip=np.array([[0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]])
res=pipeline.predict(ip)
print(res)


# In[ ]:




