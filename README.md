# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Step 1: Read the employee data from a CSV file.

Step 2: Check for null values and encode categorical variables.

Step 3: Define the features (X) and target (y).

Step 4: Split the data into training and testing sets.

Step 5: Train the Decision Tree Classifier.

step 6: Make predictions on the test data.

step 7: Calculate the accuracy of the model.

step 8: Use the model to predict new data.
```
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
