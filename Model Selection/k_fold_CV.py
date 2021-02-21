# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:49:02 2021

@author: mrtkr
"""

from sklearn.datasets import load_iris
import pandas as pd 
import numpy as np

iris = load_iris()

x = iris.data
y = iris.target

#%%normalization
x = (x - np.min(x))/(np.max(x)-np.min(x))

#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#%%knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)

#%%K fold
from sklearn.model_selection import cross_val_score
acuuracies = cross_val_score(estimator = knn,X=x_train,y=y_train,cv=10)
print("Average Accuracy :",np.mean(acuuracies))

#%%
knn.fit(x_train,y_train)
print("Test acuuracy:",knn.score(x_test,y_test))

#%%grid_search validation
from sklearn.model_selection import GridSearchCV
grid = {"n_neighbors":np.arange(1,50)}
knn2 = KNeighborsClassifier()

knn_cv =GridSearchCV(knn2,grid,cv=10)
knn_cv.fit( x,y)

#%%K değeri
print("Hyper Parameter :",knn_cv.best_params_)
print("The Best Accuracy:",knn_cv.best_score_)

#%%grid search with linear regression
x= x[:100,:]
y = y[:100:]

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)

