# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("random+forest+regression+dataset.csv",sep = ";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% Decision tree
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)

print("Predict of 7.2 point = ",rf.predict([[7.2]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

#%%
y_head2 = rf.predict(x)
from sklearn.metrics import r2_score
print("Consistent = ",r2_score(y,y_head2))

#%%visualize
plt.scatter(x, y, color = "red")
plt.plot(x_,y_head,color = "green")
plt.show()