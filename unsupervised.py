import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Kmeans implementation--
X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=42)
print(X)
df = pd.DataFrame(X,columns=["feature_1","feature_2"])
print(df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
inertia = [] #WCSS
k_range = range(1,11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
print(inertia)
# plt.plot(k_range,inertia,marker="o")
# plt.show()

kmeans_final = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)
df["cluster"] = cluster_labels
print(cluster_labels)
sns.scatterplot(x=df["feature_1"],
                y=df["feature_2"],
                hue=df["cluster"],
                palette="viridis")
# plt.show()

# DB scan--
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
from sklearn.cluster import KMeans, DBSCAN
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df["kmeans_cluster"] = kmeans_labels
sns.scatterplot(x=df["feature_1"],
                y=df["feature_2"],
                hue=df["kmeans_cluster"],
                palette="tab10")
# plt.show()

# Initialize and fit DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
df["dbscan_cluster"] = dbscan_labels

sns.scatterplot(x=df["feature_1"],
                y=df["feature_2"],
                hue=df["dbscan_cluster"],
                palette="tab10")
# plt.show()

# dimensionality reduction--
from sklearn.decomposition import PCA
X, y_true = make_blobs(n_samples=500, n_features=5 ,centers=3, cluster_std=0.60, random_state=42)
print(X)

#scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

#pca
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca)
df_pca = pd.DataFrame(X_pca, columns=["pc1","pc2"])

print("X_scaled:", X_scaled.shape)
print("y:", len(y))
print("X_pca:", X_pca.shape)
print("df_pca:", df_pca.shape)

df_pca["label"] = cluster_labels
sns.scatterplot(x=df_pca["pc1"],
                y=df_pca["pc2"],
                hue=df_pca["label"],
                palette="Set2")
plt.show()

print("Unique labels:", set(cluster_labels))
print("Total clusters:", len(set(cluster_labels)))