# -*- coding: utf-8 -*- 
""" 
Created on Wed Apr  3 14:58:26 2024 
@author: jhs70 
""" 

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn import tree 
from sklearn import metrics 

X = np.linspace(-5,5,100) 
y = np.exp(X) 

# Split the dataset into 80% training and 20% testing sets 
test_size = 0.2 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size) 

# scatter plot for overall dataset, training dataset and testing dataset 
plt.scatter(X, y, edgecolors='k', marker='o') 
plt.legend(["Overall_dataset"]) 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.title('Overall DataSet Display') 
plt.show() 

plt.scatter(x_train, y_train, edgecolors='g', marker='*') 
plt.scatter(x_test, y_test, edgecolors='r', marker='^') 
plt.legend(["Training_dataset", "Testing_dataset"]) 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.title('Training and Testing DataSet Display') 
plt.show() 

# Fitting Decision Tree Regression to the dataset 
clf = tree.DecisionTreeRegressor() 
clf = clf.fit(x_train.reshape(-1,1), y_train) 
tree.plot_tree(clf) 
plt.show() 

y_pred = clf.predict(x_test.reshape(-1,1)) 

print("predicted value") 
print(y_pred) 
print("true value") 
print(y_test) 

# RMSE for testing data set 
rmse_test = (np.sqrt(metrics.mean_squared_error(y_test,y_pred))) 
print('RMSE for Testing Decision Tree Regressor =') 
print(rmse_test) 

# RMSE for taining data set 
rmse_train = (np.sqrt(metrics.mean_squared_error(y_train,clf.predict(x_train.reshape(-1,1))))) 
print('RMSE for Training Linear Regression with Closed Form Solution =') 
print(rmse_train) 

# plot of testing and predicted dataset points 
plt.scatter(x_test, y_test, marker='o') 
plt.scatter(x_test, y_pred, marker='^', edgecolors='r') 
plt.legend(["Testing_dataset", "Predictioned_dataset"]) 
plt.xlabel('Feature 3') 
plt.ylabel('Fetrature 4') 
plt.title('Testing and Predicted Dataset Display') 
plt.show()
