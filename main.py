import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns
heart_data=pd.read_csv("heart.csv")
#first 5 rows data
print(heart_data.head(5))
#shape(rows,columns)
print(heart_data.shape)
#dataset information
print(heart_data.info())
#statistical summary
print(heart_data.describe())
# checking missing value
print(heart_data.isnull())
# count if null values are present
print(heart_data.isnull().sum())
# dataset contain duplicates so remove it 
heart_data = heart_data.drop_duplicates(keep='first')
print(heart_data.shape)
# index should be set as well as we remove duplicates those index were wrong for now
heart_data=heart_data.reset_index(drop=True)
print(heart_data)

#target duistribution
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=heart_data)
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Number of Patients")

plt.show()
#age distribution
plt.figure(figsize=(8,5))
sns.histplot(heart_data["age"], bins=15)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
#heart disease by gender
plt.figure(figsize=(6,4))
sns.countplot(x="sex", hue="target", data=heart_data)
plt.title("Heart Disease by Gender")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Count")

plt.show()



#chest pain vs heart disease

plt.figure(figsize=(7,5))
sns.countplot(data=heart_data,x="cp",hue="target")
plt.title("Chest pain vs Heart Disease")
plt.show()

# corelation heatmap finding most affecting and match between two variables

plt.figure(figsize=(12,8))

sns.heatmap(data=heart_data.corr(),annot=True,fmt=".2f",cmap="coolwarm")
plt.title("Corelation Heatmap")
plt.show()
#now it is a time to select features which are input all information required to make prediction so remove target that is the output of our project as we are predicting  heart disease which will be analyzed by target column
X = heart_data.drop("target", axis=1)
Y = heart_data["target"]

X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_Train, Y_Train)

Prediction = model.predict(X_Test)

# accuracy check
accuracyscore = accuracy_score(Y_Test, Prediction)
print("Accuracy is:", accuracyscore)

# confusion matrix
cm = confusion_matrix(Y_Test, Prediction)
print(cm)


#visual confusion matrix
plt.figure(figsize=(6,5))

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("actual")
plt.title("Confusion Matrix")



plt.show()


#classification report
print("Classification report :",classification_report(Y_Test,Prediction))
# feature importance
feature_importance=pd.DataFrame({"feature:":X.columns,"Importance":model.coef_[0]})
feature_importance=feature_importance.sort_values(by="Importance",ascending=False)
print(feature_importance)
#Roc curve

Y_probability=model.predict_proba(X_Test)[:,1]
fpr,tpr,_=roc_curve(Y_Test,Y_probability)
roc_auc=auc(fpr,tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc Curve")
plt.legend()
plt.show()
