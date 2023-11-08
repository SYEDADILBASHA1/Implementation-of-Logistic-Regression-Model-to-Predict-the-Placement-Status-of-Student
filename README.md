# EX-04 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
# DATE:19.09.2023

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation:
>Import the necessary libraries (e.g., Python, scikit-learn).
>Load and preprocess your dataset. This includes handling missing data, encoding categorical variables (if any), and splitting the data into training and testing sets.
2. Feature Selection (Optional):
>Analyze the importance of features using techniques like feature selection or feature importance scores.
>Select the most relevant features to include in your model.
3. Logistic Regression Model:
>Create a logistic regression model using scikit-learn or any other suitable library.
>Train the model on the training dataset.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SYED ADIL BASHA
RegisterNumber:  212221043008

import pandas as pd
data=pd.read_csv("/content/Placement_Data (3).csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:

# 1.Placement Data:
![1](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/9f9d0655-f38d-4c3c-b110-a809a3dcb23c)

# 2.Salary Data:
![2](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/d4e86476-9804-4837-b606-7f402dcc6b69)

# 3.checking the null() function:
![3](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/d43e916b-aec1-4eb2-9525-623d2707631b)

# 4.Data Duplicate:
![4](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/600dff54-7d35-4c96-aab2-1412968602c2)

# 5.Print Data:
![5](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/29a8e950-aa24-4985-9020-cee7e45edaf4)

# 6.Data-Status:
![6](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/26d9bbcd-b54a-4f50-9f10-538c25ad8b2d)

# 7.y_prediction array:
![7](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/4664fa57-0cb2-4e9e-b86b-f92515991b10)

# 8.Accuracy Value
![8](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/ece224f0-b35e-4bcd-b836-7176d091823a)

# 9.Confusion Array:
![9](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/f6b6918f-7c15-4d60-951b-dfa977dce8eb)

# 10.Classification Report:
![10](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/0a3e8f8f-10c6-44e2-99d0-6ec78bce075c)

# 11.Prediction of LR:
![11](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/d04a804c-269a-495b-afb2-656aeceb4c9d)

![12](https://github.com/SYEDADILBASHA1/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134796157/80be0a49-6249-4466-a4f8-e835ae1d9ec9)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
