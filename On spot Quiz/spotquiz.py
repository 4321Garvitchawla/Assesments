# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/content/DSAI-LVA-DATASET for Quiz.csv")

imputer = SimpleImputer(strategy='mean')
df['StudyTime'] = imputer.fit_transform(df[['StudyTime']])
df['PreviousTestScore'] = imputer.fit_transform(df[['PreviousTestScore']])

outlier_detector = IsolationForest(contamination=0.05)
outliers = outlier_detector.fit_predict(df[['StudyTime', 'PreviousTestScore']])
df['Outlier'] = outliers
df = df[df['Outlier'] == 1]

def categorize_pass(score):
    if score <= 60:
        return 'Fail'
    elif score <= 80:
        return 'Pass with Low Grade'
    else:
        return 'Pass with High Grade'

college_rows = df[df['ParentEducation'] == 'College']
np.random.seed(42)
college_rows['ParentEducation'] = np.random.choice(['UG', 'PG'], size=len(college_rows))

df.loc[college_rows.index] = college_rows

df['Pass'] = df['PreviousTestScore'].apply(categorize_pass)

label_encoder = LabelEncoder()
df['ParentEducation'] = label_encoder.fit_transform(df['ParentEducation'])
df['Pass'] = label_encoder.fit_transform(df['Pass'])

X = df.drop('Pass', axis=1)
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('train_data.csv', index=False)

test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('test_data.csv', index=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

rf_classifier = RandomForestClassifier(random_state=42)
svm_classifier = SVC(random_state=42)
xgb_classifier = XGBClassifier(random_state=42)

rf_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
xgb_classifier.fit(X_train, y_train)

rf_predictions = rf_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
xgb_predictions = xgb_classifier.predict(X_test)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = ['Random Forest', 'SVM', 'XGBoost']
predictions = [rf_predictions, svm_predictions, xgb_predictions]
true_labels = y_test

accuracy = []
precision = []
recall = []
f1 = []
roc_auc = []

for pred in predictions:
    accuracy.append(accuracy_score(true_labels, pred))
    precision.append(precision_score(true_labels, pred, average='weighted'))
    recall.append(recall_score(true_labels, pred, average='weighted'))
    f1.append(f1_score(true_labels, pred, average='weighted'))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
x = np.arange(len(models))

plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.bar(x, accuracy, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.ylim(0.85, 1)
plt.xticks(x, models)

plt.subplot(2, 3, 2)
plt.bar(x, precision, color='salmon')
plt.xlabel('Models')
plt.ylabel('Precision')
plt.title('Model Precision')
plt.ylim(0.85, 1)
plt.xticks(x, models)

plt.subplot(2, 3, 3)
plt.bar(x, recall, color='lightgreen')
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Model Recall')
plt.ylim(0.85, 1)
plt.xticks(x, models)

plt.subplot(2, 3, 4)
plt.bar(x, f1, color='gold')
plt.xlabel('Models')
plt.ylabel('F1-score')
plt.title('Model F1-score')
plt.ylim(0.85, 1)
plt.xticks(x, models)

plt.tight_layout()
plt.show()

