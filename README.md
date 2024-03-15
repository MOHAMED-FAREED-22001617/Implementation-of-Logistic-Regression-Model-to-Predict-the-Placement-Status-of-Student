# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
Developed by : Mohamed Fareed F
RegisterNumber : 212222230082
```
```py
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
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
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
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
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Placement_data:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/fe66412d-9902-4f22-81e9-ac7ed05a6b23)
### Salary_data:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/36fb00d6-6c56-4349-9950-2c68903fa383)
### ISNULL():
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/0db8753c-606e-4697-8461-b4fb64fdac11)
### DUPLICATED():
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/1e54ad97-2324-449c-bf08-56b5f0a3e8f6)
### Print Data:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/d0c5ab5e-ef1f-41ea-b8fd-a40f7218d608)
### iloc[:,:-1]:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/d257429d-909b-498d-b261-b438832c8732)
### Data_Status:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/9d48012d-a6b6-4d0f-ae4c-ca14ae8dd102)
### Y_Prediction array:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/04d0d3f3-487d-4524-bba2-ea17c51591d2)
### Accuray value:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/6ecc12b6-0e7f-4a2e-98f5-41c1d12a12f9)
### Confusion Array:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/1936e630-e290-4558-9237-b1a60420b503)
### Classification report:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/61d4849c-cadd-43af-acf7-2f8a2ed722fb)
### Prediction of LR:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/4689555e-1a0d-4355-a669-ab28e9a88e29)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
