# -*- coding: utf-8 -*-
"""Lvadsusr86_garvit_lab3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1622h6uYM6ex5SRLxs9dkNjWtodkyKaWZ
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, f1_score, recall_score, confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch4-Assessment/main/seeds.csv')

df

df.isnull().sum()

filling_values=df.fillna(df.mean(), inplace=True)
print("Fillng the missing value:",filling_values)

df.isnull().sum()

# Normalize the numerical columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Summary statistics
print(data.describe())

# Histogram of numerical features
data.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

#Removing Outliers
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
scaled_data = scaler.fit_transform(data[['Area',	'Perimeter',	'Compactness','Length of kernel',	'Width of kernel'	,'Asymmetry coefficient',	'Length of kernel groove']])

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

#clustering algorithm application
optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_data)

cluster_labels = kmeans.predict(scaled_data)

silhouette_avg = silhouette_score(scaled_data, cluster_labels)
print("Average silhouette score: ",silhouette_avg)

cluster_means = data.groupby('Cluster').mean()
cluster_counts = data['Cluster'].value_counts()

print("Cluster Means:")
print(cluster_means)
print("\nCluster Counts:")
print(cluster_counts)