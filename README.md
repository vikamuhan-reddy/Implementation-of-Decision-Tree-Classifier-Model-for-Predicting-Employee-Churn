# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Start with the entire dataset.
2. Select the best feature to split the data.
3. Split the dataset into subsets based on the selected feature.
4. Recursively apply steps 2 and 3 to each subset until a stopping condition is met.
5. All data points in a subset belong to the same class.
6. No more features to split on.
7. Maximum tree depth is reached.
8. Minimum number of samples in a node is reached.
9. Assign the majority class of the leaf node as the predicted class.
10. Prune the tree if necessary to prevent overfitting.
11. Evaluate the model's performance using metrics like accuracy or F1-score.

## Program:
```py
'''
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vikamuhan reddy.n
RegisterNumber: 212223240181
'''
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()    #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### data.head()
![image](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144928933/20373c74-1ddb-4e99-b9a9-f9031b09a1ce)

### data.info()
![image](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144928933/13a23a40-7b68-4f85-9828-077573a8c443)

### data.isnull().sum()
![image](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144928933/d9e5f8a6-f06f-47e2-8c0f-bb744f352fcb)

### data.head() - salary
![image](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144928933/363c89ac-a61a-4007-b0b4-40ffe1a96e9c)

### x.head()
![image](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144928933/d78ddfa4-03dc-4213-ac4d-9475509c937b)

### accuracy
![image](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144928933/6c77f3ef-84e3-4d6a-b5e8-0f0413d5e42f)

### predicted value
![image](https://github.com/vikamuhan-reddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144928933/f7d400ef-02dc-48bd-8321-2dae3f6e08b1)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
