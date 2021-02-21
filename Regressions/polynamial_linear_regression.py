# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:27:03 2021

@author: mrtkr
"""

import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv",sep=";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x, y)
plt.xlabel("araba_hiz")
plt.ylabel("araba_max_fiyat")
#plt.show()

#%% Linear Regerssion => y = b0 +b1*x
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%% Predict 
y_head = lr.predict(x)
#plt.plot(x, y_head,color = "red")


#%% Polynamial Regression => y= b0 +b1*x+b2*x^2+b3*x^3
# we can not use linear regression because our data not linear
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 2)
x_pol = pr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_pol,y)

y_head = lr2.predict(x_pol)

plt.plot(x,y_head)
plt.show()



