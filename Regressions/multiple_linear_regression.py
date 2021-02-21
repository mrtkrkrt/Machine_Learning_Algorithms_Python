# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:07:16 2021

@author: mrtkr
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("original.csv",sep=";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

multiple_linear_regresssion = LinearRegression()
multiple_linear_regresssion.fit(x, y)

print("b0: ",multiple_linear_regresssion.intercept_)
print("b1 ve b2:",multiple_linear_regresssion.coef_)

multiple_linear_regresssion.predict(np.array([[5,35],[10,35]]))
