# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Gather information and presence of null in the dataset.
4. From sklearn.tree import DecisionTreeRegressor and fir the model. 5.Find the mean square error and r squared score value of the model.
5. Check the trained model. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SAKTHI AISHWARYA.S
RegisterNumber:  212219040132
*/
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x= data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2= metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
### Initial Dataset:
![image](https://user-images.githubusercontent.com/67967960/175046433-8a67f69c-1e19-4be5-b52d-95e27f308932.png)
### Dataset information:
![image](https://user-images.githubusercontent.com/67967960/175046589-4539c61f-e03a-48ef-89a8-e522db4c87a1.png)
### Null dataset:
![image](https://user-images.githubusercontent.com/67967960/175046651-f07de90b-eab7-459f-8413-fe47be29fb39.png)
### Encoded Dataset:
![image](https://user-images.githubusercontent.com/67967960/175046715-09dedfc9-51b1-42af-88bd-d21de010f155.png)
### Mean Square Error value:
![image](https://user-images.githubusercontent.com/67967960/175046784-0ed14c69-c27a-42fc-8e07-c77896467347.png)
### R squared score:
![image](https://user-images.githubusercontent.com/67967960/175046842-26ee1d28-e7fc-4b7c-b18d-b608792e01b7.png)
### Predicted value:
![image](https://user-images.githubusercontent.com/67967960/175046989-b54d7e18-afcb-47fa-9067-a37ce89c610e.png)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
