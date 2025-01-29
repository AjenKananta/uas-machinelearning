import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load('random_forest_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')

st.title("Credit Risk Classification App")
st.write("Aplikasi ini memprediksi risiko kredit (Good/Bad Risk) menggunakan Random Forest dan Decision Tree.")

st.sidebar.header("Input Data")
st.sidebar.write("Masukkan data secara manual:")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
income = st.sidebar.number_input("Income", min_value=0, step=500)

home_ownership = st.sidebar.selectbox("Home Ownership", options=['RENT', 'OWN', 'MORTGAGE'])
home_ownership_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2}
home_ownership = home_ownership_map[home_ownership]

emp_length = st.sidebar.number_input("Employment Length (years)", min_value=0, max_value=50, step=1)

loan_intent = st.sidebar.selectbox("Loan Intent", options=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_intent_map = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
loan_intent = loan_intent_map[loan_intent]

loan_grade = st.sidebar.selectbox("Loan Grade", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
loan_grade = loan_grade_map[loan_grade]

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=100)
loan_interest_rate = st.sidebar.number_input("Loan Interest Rate", min_value=0.0, step=0.01)
loan_percent_income = st.sidebar.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, step=0.01)

default_on_file = st.sidebar.selectbox("Default on File", options=['Y', 'N'])
default_on_file_map = {'Y': 1, 'N': 0}
default_on_file = default_on_file_map[default_on_file]

credit_hist_length = st.sidebar.number_input("Credit History Length (months)", min_value=0, step=1)

input_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_home_ownership': [home_ownership],
    'person_emp_length': [emp_length],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amount],
    'loan_int_rate': [loan_interest_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [default_on_file],
    'cb_person_cred_hist_length': [credit_hist_length]
})

st.write("Data yang dimasukkan:")
st.write(input_data)

st.sidebar.header("Pilih Model")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Decision Tree"])

if st.sidebar.button("Predict"):
    model = rf_model if model_choice == "Random Forest" else dt_model  # Gunakan model yang dipilih

    prediction = model.predict(input_data)
    input_data['Prediction'] = ["Good Risk" if p == 1 else "Bad Risk" for p in prediction]

    st.write("Hasil Prediksi:")
    st.write(input_data)

st.sidebar.info("""
- Random Forest: Model berbasis pohon keputusan dengan ensemble learning.
- Decision Tree: Model berbasis pohon keputusan yang lebih sederhana dibanding Random Forest.
""")
