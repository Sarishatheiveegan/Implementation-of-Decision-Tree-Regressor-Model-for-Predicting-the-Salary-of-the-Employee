# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. imprt required python libraries and load the CSV file using pandas
2. necessary preprocessing steps to be conducted
3. use labelencoder to encode the values
4. split the data set for training and test using train_test_split
5. import DecisionTreeRegressor to make decision
6. calculate the mean squared error and R2_score
7. do the prediction on the unsean values

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MARINO SARISHA T
RegisterNumber:  212223240084
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
print("First five rows:\n",data.head())
data.info()
print("Null Values:\n",data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print("Encoded Values:\n",data.head())

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
ypred=dt.predict(xtest)

from sklearn import metrics
mse=metrics.mean_squared_error(ytest,ypred)
print("Mean squared error: ",mse)

r2=metrics.r2_score(ytest,ypred)
print("R2_Score: ",r2)

print("Prediction: ",dt.predict([[5,6]]))
```

## Output:
![Screenshot 2024-11-17 150730](https://github.com/user-attachments/assets/010d7d75-c770-40ac-908c-268716cd5244)
![Screenshot 2024-11-17 150745](https://github.com/user-attachments/assets/9308a59b-c6f2-43f2-bbdd-929130b8a728)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
