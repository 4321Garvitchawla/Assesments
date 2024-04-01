# -*- coding: utf-8 -*-
"""LVDASUSR86_garvit_Lab2_Internal2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FmWLMA3aLZ-aC2Px-QWiVN2UV7YiBy3s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch4-Assessment/main/Mall_Customers.csv")

df

df.shape

df.info()

df.isnull().sum()

filling_values=df.fillna(df.mean(), inplace=True)
print("Fillng the missing value:",filling_values)

df.isnull().sum()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
data = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Feature Engineering
data['Spending to Income Ratio'] = data['Spending Score (1-100)'] / data['Annual Income (k$)']

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    score = silhouette_score(scaled_data, kmeans.labels_)
    silhouette_scores.append(score)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis')
plt.show()

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_data)
data['Cluster'] = kmeans.labels_

# Cluster Analysis
cluster_means = data.groupby('Cluster').mean()
cluster_counts = data['Cluster'].value_counts()

print("Cluster Means:")
print(cluster_means)
print("\nCluster Counts:")
print(cluster_counts)

y_predicted = kmeans.fit_predict(df[['Age','Annual Income (k$)']])
y_predicted

"""Strategy Development Based on Clusters:

1. **Targeted Marketing:**
   - **Cluster 1:** Focus on budget-friendly promotions.
   - **Cluster 2:** Emphasize value and affordability.
   - **Cluster 3:** Offer exclusive deals and premium services.
   - **Cluster 4:** Encourage engagement with unique incentives.
   - **Cluster 5:** Provide cost-effective solutions and bundled offers.

2. **Customer Experience Enhancement:**
   - **Personalized Recommendations:** Tailor suggestions to cluster preferences.
   - **Store Layout Optimization:** Adjust layouts to match cluster behaviors.
   - **Communication Channels:** Utilize suitable channels for targeted messaging.
   - **Feedback Collection:** Gather insights for ongoing improvement.

By implementing these strategies, the retail company can enhance customer satisfaction and engagement, leading to increased loyalty and sales.
"""