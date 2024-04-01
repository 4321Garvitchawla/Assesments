# -*- coding: utf-8 -*-
"""LVDASUSR86_garvit_Lab1_Internal2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mVmfm1FNP-pTtpYUVbKiHiixLBVCBJ8T
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch4-Assessment/main/winequality-red.csv")

df

df.shape

df.info()

df.head()

#checking the null values
df.isnull().sum()

filling_values=df.fillna(df.mean(), inplace=True)
print("Fillng the missing value:",filling_values)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
data = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

#boxplot after removing the outlier
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_no_outliers)
plt.title("Boxplot of Numerical Features (Outliers Removed)")
plt.xticks(rotation=45)
plt.show()

def map_quality(quality):
    if quality in range(3, 7):
        return "Bad"
    elif quality in range(7, 9):
        return "Good"

df['quality_category'] = df['quality'].apply(map_quality)
df['quality_category'] = df['quality_category'].map({"Bad": 0, "Good": 1})

print(df)

#feature selection and data cleaning
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

df = df.drop_duplicates() #removing duplicates

df.corr()        ## geting correlation

df = df.drop(columns=['residual sugar','total sulfur dioxide','free sulfur dioxide'])

from sklearn.model_selection import train_test_split
X = df.drop(columns=['alcohol','sulphates','citric acid','volatile acidity'])
y = df['quality_category']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()