import matplotlib.pyplot as plt
import matplotlib as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('S.Y Maths MP 2023-24.csv')

df = load_data()

# Data preprocessing
m10 = df['10th']
m12 = df['12th']
mCET = df['CET']

# Handling categorical data
label_encoder = LabelEncoder()
df['FYFH Result'] = label_encoder.fit_transform(df['FYFH Result'])  # Pass-1, Fail-0

# Drop Timestamp column
df.drop('Timestamp', axis=1, inplace=True)

# Streamlit interface
st.title('Data Exploration')

# Display the DataFrame
st.write("DataFrame:", df)

# Display summary statistics
st.write("Summary Statistics:", df.describe())



# Visualizations
sem1 = df['FYFH Result']
pass_indices = [i for i, result in enumerate(sem1) if result == 1]
fail_indices = [i for i, result in enumerate(sem1) if result == 0]

# Streamlit interface
st.title('Semester 1 Performance Analysis - Data Representation')

# Plotting for 10th Board Marks
st.subheader('10th Board Marks vs. Semester 1 Performance')
fig, ax = plt.subplots()
ax.bar([m10[i] for i in pass_indices], [sem1[i]+1 for i in pass_indices], label="Pass", color='g')
ax.bar([m10[i] for i in fail_indices], [sem1[i]+1 for i in fail_indices], label="Fail", color='r')
ax.set_xlabel('10th Board Marks')
ax.set_ylabel('Semester 1 Marks')
ax.legend()
st.pyplot(fig)

# Plotting for 12th Board Marks
st.subheader('12th Board Marks vs. Semester 1 Performance')
fig, ax = plt.subplots()
ax.bar([m12[i] for i in pass_indices], [sem1[i]+1 for i in pass_indices], label="Pass", color='g')
ax.bar([m12[i] for i in fail_indices], [sem1[i]+1 for i in fail_indices], label="Fail", color='r')
ax.set_xlabel('12th Board Marks')
ax.set_ylabel('Semester 1 Marks')
ax.legend()
st.pyplot(fig)

# Plotting for CET Marks
st.subheader('CET Marks vs. Semester 1 Performance')
fig, ax = plt.subplots()
ax.bar([mCET[i] for i in pass_indices], [sem1[i]+1 for i in pass_indices], label="Pass", color='g')
ax.bar([mCET[i] for i in fail_indices], [sem1[i]+1 for i in fail_indices], label="Fail", color='r')
ax.set_xlabel('CET Marks')
ax.set_ylabel('Semester 1 Marks')
ax.legend()
st.pyplot(fig)




# Model Training
X = df[['10th', '12th', 'CET']]
y = df['FYFH Result']

# Streamlit interface
st.title('Data Preprocessing')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display scaled data
st.write("X_train_scaled:", X_train_scaled)
st.write("X_test_scaled:", X_test_scaled)
st.write("y_train:", y_train)
st.write("y_test:", y_test)


# Streamlit interface
st.title('Model Training - Random Forest')

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display evaluation results
st.write("Accuracy:", accuracy)
st.write("\nConfusion Matrix:")
st.write(conf_matrix)
st.write("\nClassification Report:")
st.write(class_report)




# Model Feedback
# Calculate feature importance
feature_importance = model.feature_importances_

# Streamlit interface
st.title('Feature Importance')

# Plot feature importance
fig, ax = plt.subplots()
ax.bar(X.columns, feature_importance)
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance')
st.pyplot(fig)




# Prediction
# Streamlit interface
st.title('Maths MP - FY Result Prediction System')

user_10th_marks = st.number_input("Enter 10th-grade marks:", min_value=0.0, max_value=100.0, step=0.1)
user_12th_marks = st.number_input("Enter 12th-grade marks:", min_value=0.0, max_value=100.0, step=0.1)
user_CET_marks = st.number_input("Enter CET marks:", min_value=0.0, max_value=100.0, step=0.1)

if st.button('Predict'):
    user_input_scaled = scaler.transform([[user_10th_marks, user_12th_marks, user_CET_marks]])
    
    # Make prediction
    prediction = model.predict(user_input_scaled)
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    st.write(f'Predicted Result: {prediction_label}')

    # Display prediction explanation
    st.write(f'With 10th marks {user_10th_marks}, 12th marks {user_12th_marks}, and CET marks {user_CET_marks}, the predicted result for the first year of engineering is: {prediction_label}')
