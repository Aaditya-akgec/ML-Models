import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("student.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Performance"] = le.fit_transform(df["Performance"])
print(df.isnull().sum())
print("duplicates = ",df.duplicated().sum())
print(df["Performance"].unique())
print(df["Performance"].value_counts())
df["Student_ID"] = df["Student_ID"].str.replace("STU", "", regex=False).astype(int)
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import StackingClassifier

X = df.drop("Performance",axis=1)
y = df["Performance"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
    
models = ({
    "logistic regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision tree": DecisionTreeClassifier(),
    "SVM": SVC()
})
result = []

for name,model in models.items():
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    result.append({
        "model": name,
        "accuracy": round(acc,4),
        "f1_score": round(f1,4,)
    })
print(result)

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=9)
model_knn.fit(X_train,y_train)
print(model_knn.score(X_test,y_test))

from sklearn.svm import SVC
model_svm = SVC(kernel= "linear",gamma="auto")
model_svm.fit(X_train,y_train)
print(model_svm.score(X_test,y_test))

#now lets use the grid search cv:--
from sklearn.model_selection import GridSearchCV
classifier = GridSearchCV((model_svm),{
    "C" : [1,10,20,30],
    "kernel" : ["rbf","linear"]
},cv = 5,return_train_score = False)
classifier.fit(X,y)
print(classifier.cv_results_)
result = pd.DataFrame(classifier.cv_results_)
print(result)
print(result[["param_C","param_kernel","mean_test_score"]])

# randomized search cv 
from sklearn.model_selection import RandomizedSearchCV
classifier_r = RandomizedSearchCV((model_svm),{
    "C" : [1,10,20,30],
    "kernel" : ["rbf","linear"]
},n_iter = 4,cv = 5,return_train_score = False)
classifier_r.fit(X,y)
results = pd.DataFrame(classifier_r.cv_results_)
print(results)
print(results[["param_C","param_kernel","mean_test_score"]])
