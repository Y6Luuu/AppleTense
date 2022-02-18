import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# read_csv
df = pd.read_csv(r'C:\Users\luuy\Desktop\DT_pretest_YL_20220208.csv')
X = df[['Brix', 'FS', 'AFW', 'JU', 'FC', 'RFW', 'RT', 'DP', 'SL', 'ER', 'HR']]
y = df[['MALO']]
df.describe()

target_column = ['MALO'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(X_train, y_train) 
pred_train_rf= model_rf.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rf)))
print(r2_score(y_train, pred_train_rf))

pred_test_rf = model_rf.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(y_test, pred_test_rf))

from sklearn import metrics

print('ev:', metrics.explained_variance_score(y_test,pred_test_rf))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,pred_test_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,pred_test_rf))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test,pred_test_rf)))

pre_y_list = [] 
pre_y_list.append(model_rf.fit(X, y).predict(X)) 

# Visualization
plt.figure()  
plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  
color_list = ['r', 'b', 'g', 'y', 'c']  
linestyle_list = ['-', '.', 'o', 'v', '*']  
for i, pre_y in enumerate(pre_y_list): 
    plt.plot(np.arange(X.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  
plt.title('regression result comparison')  
plt.legend(loc='upper right')  
plt.ylabel('real and predicted value')  
plt.show()  

