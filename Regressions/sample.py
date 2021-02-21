# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:48:16 2021

@author: mrtkr
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("column_2C_weka.csv")
data = df[df["class"] == "Abnormal"]

x = np.array(data.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data.loc[:,'sacral_slope']).reshape(-1,1)

#%% linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

p_space = np.arange(min(x),max(x),0.01).reshape(-1,1)

reg.fit(x,y)

y_head = reg.predict(p_space)

print("Consistent Linear Regression = ",reg.score(x, y))
plt.plot(p_space,y_head,color="red",linewidth = 3)

plt.scatter(x, y,color = "green")


#%%polynamial regression
from sklearn.tree import DecisionTreeRegressor
tr = DecisionTreeRegressor()

tr.fit(x,y)

y_head_tree = tr.predict(p_space)
print("Consisten Decision Tree Regression",tr.score(x,y))
plt.plot(p_space,y_head_tree,color="black",linewidth=3)

#%%random forest regression
from sklearn.ensemble import RandomForestRegressor

fr = RandomForestRegressor(n_estimators=100,random_state=42)

fr.fit(x,y)
y_head_forest = fr.predict(p_space)

print("Consistent Random Forest Regression  = ",fr.score(x, y))
plt.plot(p_space,y_head_forest,color = "yellow",linewidth = 4)