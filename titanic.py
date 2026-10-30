import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
df = sns.load_dataset("titanic")
print(df)
print(df.columns)
print(df.info())
df.drop(['class', 'who', 'adult_male', 'deck', 'embark_town',
       'alive'],axis=1,inplace = True)
print(df.info())
                           # data cleaning and handling--
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
print(df.head())
df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"])
print(df.head())
print(df.loc[5])
df = df.dropna().astype(int) 
print(df.head(10))
                     # model training & prediction 
X = df.drop('survived', axis = 1)
y = df['survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
               
y_pred = model.predict(X_test)
print(y_pred)
print(y_test) 
                 # confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# knn 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
print(y_pred_knn)
print(accuracy_score(y_test,y_pred_knn))
print(confusion_matrix(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn))

#naive bayes--
from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train,y_train)
y_pred_nb = model_nb.predict(X_test)
print(y_pred_nb)
print(accuracy_score(y_test,y_pred_nb))
print(confusion_matrix(y_test,y_pred_nb))
print(classification_report(y_test,y_pred_nb))

#decision trees--
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_scaled, y_train)
y_pred_dt = model_dt.predict(X_test_scaled)
print(y_pred_dt)
print(accuracy_score(y_test,y_pred_dt))
print(confusion_matrix(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))

#SVM
from sklearn.svm import SVC
model_svm = SVC(kernel='rbf')
model_svm.fit(X_train_scaled, y_train)
y_pred_svc = model_svm.predict(X_test_scaled)
print(y_pred_svc)
print(accuracy_score(y_test,y_pred_svc))
print(confusion_matrix(y_test,y_pred_svc))
print(classification_report(y_test,y_pred_svc))

#cross validation--
print(df)
X = df.drop("survived",axis=1)
y = df["survived"]
from sklearn.model_selection import cross_val_score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(model_svm,X_scaled,y,cv=5,scoring='accuracy')
print(scores)
print(scores.mean())
