import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             roc_curve, auc, mean_squared_error)
import joblib

# Streamlit Title
st.title("Disease Prediction and Risk Analysis")

# 1. Synthetic Dataset Generation
@st.cache_data
def generate_data(n_samples=500):
    np.random.seed(42)
    age = np.random.randint(20, 80, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    blood_pressure = np.random.normal(130, 15, n_samples).astype(int)
    cholesterol = np.random.normal(220, 30, n_samples).astype(int)

    risk_score = (
        0.3 * (age - 20) / 60 +
        0.2 * sex +
        0.25 * (blood_pressure - 90) / 90 +
        0.25 * (cholesterol - 150) / 150
    )
    risk_score = np.clip(risk_score, 0, 1)
    disease = (risk_score + np.random.normal(0, 0.1, n_samples)) > 0.5
    disease = disease.astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'disease': disease,
        'risk_score': risk_score.round(2)
    })
    return df

data = generate_data()

# 2. Show Raw Data
if st.checkbox("Show raw data"):
    st.write(data.head())

# 3. Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# 4. Model Training
X = data[['age', 'sex', 'blood_pressure', 'cholesterol']]
y_class = data['disease']
y_reg = data['risk_score']

X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_class_train)
y_class_pred = log_model.predict(X_test)
y_class_prob = log_model.predict_proba(X_test)[:, 1]

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_reg_train)
y_reg_pred = lin_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_class_train)
y_rf_pred = rf_model.predict(X_test)

# 5. Evaluation Outputs
st.subheader("Evaluation Metrics")
st.write("Logistic Regression Accuracy:", accuracy_score(y_class_test, y_class_pred))
st.write("Random Forest Accuracy:", accuracy_score(y_class_test, y_rf_pred))
st.write("Linear Regression MSE:", mean_squared_error(y_reg_test, y_reg_pred))

# Confusion Matrix
st.subheader("Confusion Matrix (Logistic Regression)")
cm = confusion_matrix(y_class_test, y_class_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Predicted Probabilities Histogram
st.subheader("Predicted Probabilities (Logistic Regression)")
fig, ax = plt.subplots()
sns.histplot(y_class_prob, bins=10, kde=True, ax=ax)
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_class_test, y_class_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# 6. Save Models and Data
joblib.dump(log_model, 'logistic_model.pkl')
joblib.dump(lin_model, 'linear_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
data.to_csv('synthetic_patient_data.csv', index=False)

# 7. Prediction Interface
st.subheader("Make a Prediction")

age_input = st.slider("Age", 20, 80, 40)
sex_input = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
bp_input = st.slider("Blood Pressure", 90, 180, 120)
chol_input = st.slider("Cholesterol", 150, 300, 200)

input_data = pd.DataFrame({
    'age': [age_input],
    'sex': [sex_input],
    'blood_pressure': [bp_input],
    'cholesterol': [chol_input]
})

# Load models from saved files
log_model = joblib.load('logistic_model.pkl')
lin_model = joblib.load('linear_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

if st.button("Predict"):
    disease_pred_log = log_model.predict(input_data)[0]
    disease_prob_log = log_model.predict_proba(input_data)[0][1]
    risk_pred = lin_model.predict(input_data)[0]
    disease_pred_rf = rf_model.predict(input_data)[0]

    st.markdown(f"*Predicted Risk Score (Linear Regression):* {risk_pred:.2f}")
    st.markdown(f"*Disease Probability (Logistic Regression):* {disease_prob_log:.2f}")
    st.markdown(f"*Disease Prediction (Logistic Regression):* {'Positive' if disease_pred_log == 1 else 'Negative'}")
    st.markdown(f"*Disease Prediction (Random Forest):* {'Positive' if disease_pred_rf == 1 else 'Negative'}")
