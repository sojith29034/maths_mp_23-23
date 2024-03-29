
"""# Dataset - import, clean and process"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


df = pd.read_csv('S.Y Maths MP 2023-24.csv')

m10 = df['10th']
m12 = df['12th']
mCET = df['CET']


# Handling categorical data - converting Pass/Fail to binary
label_encoder = LabelEncoder()
df['FYFH Result'] = label_encoder.fit_transform(df['FYFH Result'])
#Pass-1 and Fail-0

sem1 = df['FYFH Result']
df.drop('Timestamp',axis=1,inplace=True)
print(df)

df.describe()

"""# Data Representation - Bar graph"""

# bar
pass_indices = [i for i, result in enumerate(sem1) if result == 1]
fail_indices = [i for i, result in enumerate(sem1) if result == 0]

plt.bar([m10[i] for i in pass_indices], [sem1[i]+1 for i in pass_indices], label="Pass", color='g')
plt.bar([m10[i] for i in fail_indices], [sem1[i]+1 for i in fail_indices], label="Fail", color='r')
plt.xlabel('10th Board Marks')
plt.ylabel('Semester 1 Marks')
plt.legend()
plt.show()

plt.bar([m12[i] for i in pass_indices], [sem1[i]+1 for i in pass_indices], label="Pass", color='g')
plt.bar([m12[i] for i in fail_indices], [sem1[i]+1 for i in fail_indices], label="Fail", color='r')
plt.xlabel('12th Board Marks')
plt.ylabel('Semester 1 Marks')
plt.legend()
plt.show()

plt.bar([mCET[i] for i in pass_indices], [sem1[i]+1 for i in pass_indices], label="Pass", color='g')
plt.bar([mCET[i] for i in fail_indices], [sem1[i]+1 for i in fail_indices], label="Fail", color='r')
plt.xlabel('CET Marks')
plt.ylabel('Semester 1 Marks')
plt.legend()
plt.show()

"""# Data Scaling"""

X = df[['10th', '12th', 'CET']]
y = df['FYFH Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train",X_train_scaled)
print("X_test",X_test_scaled)
print("y_train",y_train)
print("y_test",y_test)

"""# Model Training - RandomForest"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""# Model Feedback"""

feature_importance = model.feature_importances_
print(feature_importance)

plt.bar(X.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

"""# Finalization"""

# Example user input (you can modify this according to your interface)
user_10th_marks = float(input("Enter 10th-grade marks: "))
user_12th_marks = float(input("Enter 12th-grade marks: "))
user_CET_marks = float(input("Enter CET marks: "))


# Assuming 'scaler' is the StandardScaler used during training
user_input_scaled = scaler.transform([[user_10th_marks, user_12th_marks, user_CET_marks]])


user_prediction = model.predict(user_input_scaled)
user_prediction_label = label_encoder.inverse_transform(user_prediction)[0]
print("Predicted Result:", user_prediction_label)


print(f"With 10th marks {user_10th_marks}, 12th marks {user_12th_marks}, and CET marks {user_CET_marks},")
print("the predicted result for the first year of engineering is: {user_prediction_label}")