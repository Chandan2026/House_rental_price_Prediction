# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:32:38 2021

@author: chandan Kumar Sahoo
"""

import pandas as pd
df=pd.read_csv('rentaldata.csv')

X=df.iloc[:,2:6].values
y=df.iloc[:,7].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.neighbors import KNeighborsRegressor
nnr=KNeighborsRegressor(n_neighbors=4)
nnr.fit(X_train,y_train)


print(nnr.score(X_test,y_test))

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_1st=[]
for k in range(1,20):
    nn_model = KNeighborsRegressor(n_neighbors=k)
    nn_model.fit(X_train,y_train)
    y_pred=nn_model.predict(X_test)
    error=sqrt(mean_squared_error(y_test,y_pred))
    rmse_1st.append(error)
    print(k,error)
    
graph= pd.DataFrame(rmse_1st)
graph.plot()