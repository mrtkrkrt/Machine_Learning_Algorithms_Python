# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:17:04 2021

@author: mrtkr
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

data.diagnosis =[1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#%%normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%train-test-split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

#%%random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

rf.fit(x_train,y_train)
print("Accuracy:",rf.score(x_test,y_test))
