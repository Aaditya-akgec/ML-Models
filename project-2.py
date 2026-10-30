import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("heart.csv")
# print(df.head())
# print(df.info())
# print(df.duplicated().sum())
# print(df["HeartDisease"].value_counts())
# print(df.isnull().sum())
# print(df["Cholesterol"].value_counts())

# def plotting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var], kde = True)

# plotting("Age",1)
# plotting("RestingBP",2)
# plotting("Cholesterol",3)
# plotting("MaxHR",4)

# plt.tight_layout()
# plt.show()
#                      #cleaning data 
# # print(df["Cholesterol"].value_counts())
# ch_mean = df.loc[df["Cholesterol"]!= 0,'Cholesterol'].mean()
# print(ch_mean)
# df["Cholesterol"] = df["Cholesterol"].replace(0,ch_mean)
# df["Cholesterol"] = df['Cholesterol'].round(2)

# ch_mean = df.loc[df["RestingBP"]!= 0,'RestingBP'].mean()
# print(ch_mean)
# df["RestingBP"] = df["RestingBP"].replace(0,ch_mean)
# df["RestingBP"] = df['RestingBP'].round(2)

# def plotting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var], kde = True)

# plotting("Age",1)
# plotting("RestingBP",2)
# plotting("Cholesterol",3)
# plotting("MaxHR",4)

# plt.tight_layout()
# plt.show()
# sns.countplot(x = df['Sex'],hue = df["HeartDisease"])
# sns.countplot(x = df['ChestPainType'], hue = df["HeartDisease"])
# sns.violinplot(x = df["HeartDisease"],y="Age", data=df)
# plt.show()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["ST_Slope"] = le.fit_transform(df["ST_Slope"])
df["RestingECG"] = le.fit_transform(df["RestingECG"])
df["ExerciseAngina"] = le.fit_transform(df["ExerciseAngina"])

df = pd.get_dummies(df, columns=["ChestPainType"], drop_first=True)

df = df.astype(int)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

X = df.drop("HeartDisease",axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

models = ({
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision tree": DecisionTreeClassifier(),
    "SVM": SVC()
})
result = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    result.append({
        'model': name,
        'Accuracy': round(acc, 4),
        'f1 score': round(f1, 4),
    })
print(result)