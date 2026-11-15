
# K-Means and Hierarchical Clustering

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Read dataset
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

print("Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 2. Select numerical features
data = df[["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP"]]

# Handle missing values
data = data.fillna(data.mean())

# 3. Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# 4. Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# 5. K-Means Clustering
k = 3  # Change according to elbow graph

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X)

print("\nK-Means Cluster Results:")
print(df["KMeans_Cluster"].value_counts().sort_index())

# 6. Visualize K-Means clusters
plt.scatter(X[:, 0], X[:, 2],
            c=df["KMeans_Cluster"], cmap="rainbow")

plt.xlabel("Quantity Ordered (Scaled)")
plt.ylabel("Sales (Scaled)")
plt.title("K-Means Clustering")
plt.show()

# 7. Hierarchical Clustering
# Create dendrogram
Z = linkage(X, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points / Merged Clusters")
plt.ylabel("Distance")
plt.show()

# 8. Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
df["Hierarchical_Cluster"] = hc.fit_predict(X)

print("\nHierarchical Cluster Results:")
print(df["Hierarchical_Cluster"].value_counts().sort_index())

# 9. Display final results
print("\nFinal Dataset:")
print(df[["SALES", "KMeans_Cluster",
          "Hierarchical_Cluster"]].head(10))
