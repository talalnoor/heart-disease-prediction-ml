import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("heart.csv")

# =========================
# 2. BASIC INFO
# =========================
print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nStatistics:")
print(df.describe())

# Missing values
print("\nMissing values:\n", df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()
df = df.reset_index(drop=True)

print("\nAfter cleaning shape:", df.shape)

# =========================
# 3. EDA (VISUALIZATION)
# =========================

plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=15)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="sex", hue="target", data=df)
plt.title("Heart Disease by Gender")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x="cp", hue="target", data=df)
plt.title("Chest Pain vs Heart Disease")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 4. SPLIT DATA
# =========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. MODEL TRAINING
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# 6. PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. EVALUATION
# =========================
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# 8. ROC CURVE
# =========================
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================
# 9. FEATURE IMPORTANCE
# =========================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(feature_importance)
