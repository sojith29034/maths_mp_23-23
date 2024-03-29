import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
"""# Data Representation - Bar graph"""
sem1 = df['FYFH Result']
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

# Model Training
# (Put your model training code here)

# Model Feedback
# (Put your model feedback code here)

# Prediction
# (Put your prediction code here)

# Streamlit interface
st.title('Maths MP Prediction')

user_10th_marks = st.number_input("Enter 10th-grade marks:", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
user_12th_marks = st.number_input("Enter 12th-grade marks:", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
user_CET_marks = st.number_input("Enter CET marks:", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

if st.button('Predict'):
    # Make prediction
    prediction = model.predict([[user_10th_marks, user_12th_marks, user_CET_marks]])
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    st.write(f'Predicted Result: {prediction_label}')

    # Display prediction explanation
    st.write(f'With 10th marks {user_10th_marks}, 12th marks {user_12th_marks}, and CET marks {user_CET_marks},')
    st.write(f'the predicted result for the first year of engineering is: {prediction_label}')
