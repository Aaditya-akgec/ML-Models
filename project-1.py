import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("insurance_.csv")
print(df)
print(df.shape)
print(df.describe)
print(df.isnull().sum())
print(df.columns)

# numeric_value = ['age', 'bmi', 'children', 'charges']
# for col in numeric_value:
#     plt.figure(figsize=(6,4))
#     sns.histplot(df[col],kde = True,bins=20)
#     plt.show()
#     print(sns.countplot(x = df['children']))
#     print(sns.countplot(x = df['sex']))
#     print(sns.countplot(x = df['smoker']))

# for col in numeric_value:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(x = df[col])
#     plt.show()

                            # data cleaning and preprocessing 
df_cleaned = df.copy()
print(df_cleaned.head())
print(df_cleaned.drop_duplicates(inplace = True))
print(df_cleaned.shape)
print(df_cleaned.isnull().sum())
print(df_cleaned.dtypes)
print(df_cleaned["sex"].value_counts())
                    #changing to numeric value
df_cleaned["sex"] = df_cleaned["sex"].map({"male":0,"female":1})     
print(df_cleaned.head())
# smoker

print(df_cleaned["smoker"].value_counts())
df_cleaned["smoker"] = df_cleaned["smoker"].map({"yes":1,"no":0})
print(df_cleaned)

print(df_cleaned.rename(columns={"sex":"is_female","smoker":"is_smoker"}, inplace=True))
print(df_cleaned)

# region
print(df["region"].value_counts())
df_cleaned = pd.get_dummies(df_cleaned,columns=["region"],drop_first=True)
print(df_cleaned)
df_cleaned = df_cleaned.astype(int)
print(df_cleaned.head())

                            # feature engineering and extraction
df_cleaned["bmi_category"] = pd.cut(
    df_cleaned["bmi"],bins=[0, 18.5 , 24.9 , 29.9 , float("inf")],
    labels=["underweight","normal","overweight","obese"]
)
print(df_cleaned.head())
df_cleaned = pd.get_dummies(df_cleaned,columns=["bmi_category"],drop_first=True)
df_cleaned = df_cleaned.astype(int)
print(df_cleaned.head())

from sklearn.preprocessing import StandardScaler
cols = ["age","bmi","children"]
scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())

correlation = df_cleaned.corr(numeric_only=True)["charges"].sort_values(ascending=False)

print(correlation)

# chai square test--
df_cleaned["charges_category"] = pd.qcut(
    df_cleaned["charges"],
    q=3,
    labels=["Low", "Medium", "High"]
)
from scipy.stats import chi2_contingency

categorical_cols = ["is_female", "is_smoker", "region_northwest",
                    "region_southeast", "region_southwest",
                    "bmi_category_normal", "bmi_category_overweight",
                    "bmi_category_obese"]

for col in categorical_cols:
    contingency_table = pd.crosstab(df_cleaned[col],
                                    df_cleaned["charges_category"])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"\nFeature: {col}")
    print(f"Chi-Square Statistic = {chi2:.3f}")
    print(f"P-value = {p:.5f}")

    if p < 0.05:
        print("Result: Significant association")
    else:
        print("Result: No significant association")

# All data--
final_df = df_cleaned[["age","is_female","is_smoker","charges"]]    
print(final_df)

from sklearn.model_selection import train_test_split

X = final_df.drop('charges', axis = 1)
y = final_df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print(r2)
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1-r2) * (n-1) / (n-p-1))
print(adjusted_r2)