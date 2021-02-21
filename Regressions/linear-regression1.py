# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import library
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#import data
df = pd.read_csv("linear_regression_dataset.csv",sep = ";")

#plot data
"""plt.scatter(df.deneyim,df.maas)
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.show()"""

#%% Lİnear Regression

#skLearn Library
from sklearn.linear_model import LinearRegression

#Linear Regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%% predict

b0 = linear_reg.predict([[0]])
print("bo: ",b0)   #intercept

b0_ = linear_reg.intercept_
print("b0_: ",b0_)   #intercept

b1 = linear_reg.coef_
print("b1: ",b1)    #slope

#maas = 1663 + 1138*deneyim

print(linear_reg.predict([[11]]))
print(linear_reg.score(x,y))

#visualize line 

array = np.array([0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16]).reshape(-1,1)

plt.scatter(x,y)

y_head = linear_reg.predict(array)
plt.plot(array,y_head,color = "red")
plt.show()

