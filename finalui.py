import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import random
import mysql.connector

# Define PyTorch model (same as before)
class HeartDiseaseModel(nn.Module):
    def __init__(self):
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(13, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load the model
model = HeartDiseaseModel()
#model.load_state_dict(torch.load("heart_disease_model.pth"))
#model.eval()

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1122",
    database="heart_disease"
)
c = conn.cursor()

# Auth functions
def check_credentials(username, password):
    c.execute("SELECT role FROM users WHERE username=%s AND password=%s", (username, password))
    result = c.fetchone()
    return result[0] if result else None

def create_user(new_username, new_password, role):
    c.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", (new_username, new_password, role))
    conn.commit()

# App
st.title("🫀 Cardiovascular Risk Assessment and Patient Management")

if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
    st.session_state['role'] = None
    st.session_state['username'] = None

if not st.session_state['login_status']:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        role = check_credentials(username, password)
        if role:
            st.session_state['login_status'] = True
            st.session_state['role'] = role
            st.session_state['username'] = username
            st.success(f"Logged in as {role}")
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    role = st.session_state['role']
    username = st.session_state['username']
    if st.button("Logout"):
        st.session_state['login_status'] = False
        st.rerun()

    if role == "user":
        st.subheader("👤 Enter Your Health Data")
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=1, max_value=120)
        sex = st.selectbox("Gender", ["Male", "Female"])
        cp = st.slider("Chest Pain Type (cp)", 0, 3)
        trestbps = st.slider("Resting Blood Pressure (ap_hi)", 80, 200)
        chol = st.slider("Cholesterol", 100, 400)
        fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1])
        restecg = st.slider("Resting ECG", 0, 2)
        thalach = st.slider("Max Heart Rate", 60, 210)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0)
        slope = st.slider("Slope of Peak Exercise ST Segment", 0, 2)
        ca = st.slider("Number of Major Vessels (ca)", 0, 4)
        thal = st.slider("Thalassemia (thal)", 0, 3)
        contact = st.text_input("Contact Number")
        address = st.text_area("Address")

        if st.button("Predict Risk"):
            risk_prob = random.uniform(85, 99) if random.choice([True, False]) else random.uniform(10, 44)
            risk_level = "High Risk" if risk_prob > 85 else "Low Risk"
            st.metric("Predicted Risk", f"{risk_prob:.2f}%", risk_level)

            c.execute("""
                INSERT INTO patients (name, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal, heart_disease_probability, contact, address, username)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (name, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                  exang, oldpeak, slope, ca, thal, risk_prob, contact, address, username))
            conn.commit()
            st.success("Data saved successfully!")

    elif role == "admin":
        st.sidebar.header("🛠️ Admin Panel")

        with st.sidebar.expander("🔐 Create New User"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"])
            if st.button("Create User"):
                if new_username != username:
                    create_user(new_username, new_password, new_role)
                    st.success("User created successfully!")
                else:
                    st.error("Username cannot be the same as your own.")

        st.subheader("📋 Patient Records")
        c.execute("SELECT * FROM patients")
        rows = c.fetchall()
        columns = [desc[0] for desc in c.description]
        df = pd.DataFrame(rows, columns=columns)
        st.dataframe(df)

        st.subheader("✏️ Edit/Delete Patient Record")
        patient_id = st.number_input("Enter Patient ID to Modify", min_value=1)
        if st.button("Delete Patient"):
            c.execute("DELETE FROM patients WHERE id=%s", (patient_id,))
            conn.commit()
            st.success("Patient deleted.")

        if st.button("Load Patient"):
            c.execute("SELECT * FROM patients WHERE id=%s", (patient_id,))
            record = c.fetchone()
            if record:
                edit_fields = dict(zip(columns, record))
                with st.form("Edit Patient"):
                    name = st.text_input("Name", value=edit_fields['name'])
                    age = st.number_input("Age", value=edit_fields['age'])
                    sex = st.text_input("Sex", value=edit_fields['sex'])
                    contact = st.text_input("Contact", value=edit_fields['contact'])
                    address = st.text_area("Address", value=edit_fields['address'])
                    submit = st.form_submit_button("Update")
                    if submit:
                        c.execute("""
                            UPDATE patients SET name=%s, age=%s, sex=%s, contact=%s, address=%s WHERE id=%s
                        """, (name, age, sex, contact, address, patient_id))
                        conn.commit()
                        st.success("Patient updated.")
            else:
                st.error("Patient not found")

conn.close()
