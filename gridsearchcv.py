import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
df = sns.load_dataset("iris")
print(df.head())
print(df["species"].unique())

from sklearn.model_selection import train_test_split
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=13)
model_knn.fit(X_train,y_train)
print(model_knn.score(X_test,y_test))

from sklearn.svm import SVC
model_svm = SVC(kernel="rbf")
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

#randomized search cv 
from sklearn.model_selection import RandomizedSearchCV
classifier_r = RandomizedSearchCV((model_svm),{
    "C" : [1,10,20,30],
    "kernel" : ["rbf","linear"]
},n_iter = 4,cv = 5,return_train_score = False)
classifier_r.fit(X,y)
results = pd.DataFrame(classifier_r.cv_results_)
print(results)
print(results[["param_C","param_kernel","mean_test_score"]])

#ensemble learning (stacking)
X = df.drop("species",axis = 1)
y = df["species"]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_encoded)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import StackingClassifier

base_learners = [
    ("dt", DecisionTreeClassifier(random_state=42)),
    ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
    ("lr", LogisticRegression(max_iter=1000))
]
meta_learner = LogisticRegression(max_iter=1000)
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
) 
stacking_clf.fit(X_train,y_train)
y_pred = stacking_clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(acc)

#bagging--
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(acc)

#boosting--
