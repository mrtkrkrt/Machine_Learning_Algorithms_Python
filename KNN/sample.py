# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:08:54 2021

@author: mrtkr
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("column_2C_weka.csv")

df.classs = [1 if i =="Abnormal" else 0 for i in df.classs]

y = df["classs"].values
x_data = df.drop(["classs"],axis=1)

#%%normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%train-test-split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 2)

#%%KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(x_train,y_train)

print("{} nn score = {}".format(3,knn.score(x_test,y_test)))

#%%findK value
score_list = []
for i in range(1,200):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,200),score_list)
plt.show()

#%%Conclusion
# The best K value is 15


