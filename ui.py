import streamlit as st
import sqlite3
import pandas as pd
import streamlit
import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import sqlite3
import numpy as np

# Define the model architecture before loading
class HeartDiseaseModel(nn.Module):
    def __init__(self):
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(13, 16)  # Input Layer
        self.fc2 = nn.Linear(16, 8)   # Hidden Layer
        self.fc3 = nn.Linear(8, 1)    # Output Layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Output between 0 and 1
        return x

# Load model after defining the class
model = HeartDiseaseModel()
model.load_state_dict(torch.load("heart_disease_model.pth"))
model.eval()




# Connect to SQLite Database (Replace with MySQL/PostgreSQL if needed)
conn = sqlite3.connect("heart_disease.db")
c = conn.cursor()

# Create Table if Not Exists
c.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    sex TEXT,
    cp INTEGER,
    trestbps INTEGER,
    chol INTEGER,
    fbs INTEGER,
    restecg INTEGER,
    thalach INTEGER,
    exang INTEGER,
    oldpeak REAL,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    heart_disease_probability REAL,
    contact TEXT,
    address TEXT
)
""")
conn.commit()

### **Function to Add Patient**
def add_patient(name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, heart_disease_probability, contact, address):
    c.execute("""
    INSERT INTO patients (name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, heart_disease_probability, contact, address)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, heart_disease_probability, contact, address))
    conn.commit()

### **Function to View Data**
def view_patients():
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    return df

### **Function to Update Patient Details**
def update_patient(id, name, age, contact, address):
    c.execute("""
    UPDATE patients
    SET name=?, age=?, contact=?, address=?
    WHERE id=?
    """, (name, age, contact, address, id))
    conn.commit()

### **Function to Delete a Patient**
def delete_patient(id):
    c.execute("DELETE FROM patients WHERE id=?", (id,))
    conn.commit()

### **Streamlit UI**
st.title("🏥 Heart Disease Prediction & Patient Database")

menu = st.sidebar.selectbox("Menu", ["Add Patient", "View Patients", "Update Patient", "Delete Patient","Project Details "])

if menu == "Add Patient":
    st.subheader("➕ Add New Patient")

    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)")
    chol = st.number_input("Cholesterol Level (chol)")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.number_input("Resting ECG (restecg)", min_value=0, max_value=2)
    thalach = st.number_input("Max Heart Rate Achieved (thalach)")
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)")
    slope = st.number_input("Slope", min_value=0, max_value=2)
    ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=3)
    thal = st.number_input("Thalassemia (thal)", min_value=0, max_value=3)
    contact = st.text_input("Contact Number")
    address = st.text_area("Address")

if st.button("Predict & Add Patient"):
    input_data = torch.tensor([[cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, age, 1 if sex == "Male" else 0]], dtype=torch.float32)

    with torch.no_grad():
        heart_disease_probability = model(input_data).item()  # Get prediction

    heart_disease_probability = round(heart_disease_probability * 100, 2)  # Convert to percentage

    add_patient(name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, heart_disease_probability, contact, address)
    st.success(f"Patient {name} added successfully! Estimated Heart Disease Probability: {heart_disease_probability}%")


elif menu == "View Patients":
    st.subheader("📋 View Patients")
    df = view_patients()
    st.dataframe(df)

elif menu == "Update Patient":
    st.subheader("✏️ Update Patient Details")
    df = view_patients()
    patient_ids = df["id"].tolist()
    
    if patient_ids:
        patient_id = st.selectbox("Select Patient ID", patient_ids)
        selected_patient = df[df["id"] == patient_id].iloc[0]

        new_name = st.text_input("Name", selected_patient["name"])
        new_age = st.number_input("Age", value=selected_patient["age"])
        new_contact = st.text_input("Contact", selected_patient["contact"])
        new_address = st.text_area("Address", selected_patient["address"])

        if st.button("Update Patient"):
            update_patient(patient_id, new_name, new_age, new_contact, new_address)
            st.success(f"Patient {new_name} updated successfully!")

elif menu == "Delete Patient":
    st.subheader("🗑️ Delete Patient")
    df = view_patients()
    patient_ids = df["id"].tolist()

    if patient_ids:
        delete_id = st.selectbox("Select Patient ID to Delete", patient_ids)

        if st.button("Delete"):
            delete_patient(delete_id)
            st.warning(f"Patient with ID {delete_id} deleted!")
elif menu == "Project Details ":
    st.title("Heart Disease Prediction System")
    
    st.markdown("### Developed by:")
    st.markdown("- **N Shiva Manohara Reddy**")
    st.markdown("- **L Tarun Aditya**")
    st.markdown("- **Mukesh Chand Y**")
    st.markdown("- **Nagarjuna**")
    
    st.markdown("### Institution:")
    st.markdown("**The Oxford College of Engineering, Bangalore**")
    
    st.markdown("### Guided by:")
    st.markdown("**Professor Maheswari Patel**")


conn.close()
