#import library
import numpy as np  
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet 
from sklearn.svm import SVR  
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.model_selection import cross_val_score  
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  
import pandas as pd 
import matplotlib.pyplot as plt  

# read_csv
data = pd.read_csv(r'C:\Users\luuy\Desktop\DT_pretest_YL_20220208.csv')

#definition
X = data[['Brix', 'FS', 'AFW', 'JU', 'FC', 'RFW', 'RT', 'DP', 'SL', 'ER', 'HR']]
y = data[['MALO']]

# Train regression models
n_folds = 10  # cross validation
model_br = BayesianRidge()  
model_lr = LinearRegression()  
model_etc = ElasticNet()  
model_svr = SVR()  
model_gbr = GradientBoostingRegressor()  
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR'] 
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr]  
cv_score_list = []  
pre_y_list = []  
for model in model_dic:  
    scores = cross_val_score(model, X, y, cv=n_folds)  
    cv_score_list.append(scores)  
    pre_y_list.append(model.fit(X, y).predict(X))  

# model evaluation
n_samples, n_features = X.shape  
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  
model_metrics_list = []  
for i in range(5):  
    tmp_list = []  
    for m in model_metrics_name:  
        tmp_score = m(y, pre_y_list[i])  
        tmp_list.append(tmp_score)  
    model_metrics_list.append(tmp_list)  
df1 = pd.DataFrame(cv_score_list, index=model_names)  
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  
print ('samples: %d \t features: %d' % (n_samples, n_features))  
print (70 * '-')  
print ('cross validation result:')  
print (df1)  
print (70 * '-')  
print ('regression metrics:')  
print (df2)  
print (70 * '-')  
print ('short name \t full name')  
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-') 

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