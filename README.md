# Length-of-stay-in-hospital-ML-Project
Part of college project

****************************************************

🛏️ Patient Length of Stay Prediction
📌 Project Overview
The healthcare sector has always embraced technological advancements to enhance patient care and hospital management. With the rise of machine learning, hospitals can now leverage data to optimize staff allocation, resource management, and patient treatment plans.

This project focuses on predicting patient length of stay (LOS) at the time of admission using machine learning techniques. Predicting LOS helps in effective hospital resource allocation, reducing patient overload, and minimizing the risk of infections.

🩺 Problem Statement
The COVID-19 pandemic highlighted the importance of efficient healthcare management, particularly in managing hospital resources and patient flow. One critical metric for hospital efficiency is patient Length of Stay (LOS)—the duration a patient stays in the hospital.

Objective:
To predict the LOS of a patient using historical hospital data, allowing hospitals to:
✅ Identify high-risk patients who may require prolonged care.
✅ Optimize treatment plans to reduce LOS.
✅ Improve logistics (bed allocation, staffing, equipment management).

LOS Categories (Target Variable):
The LOS is classified into 11 different categories, ranging from 0-10 days to more than 100 days.

📂 Dataset Overview
The dataset contains multiple features related to hospital admissions, such as:

Patient demographics (Age, Admission type)
Hospital details (Department, Ward Type, Facility Code)
Medical condition severity (Severity of Illness)
Financials (Admission Deposit)
🛠️ Tech Stack Used
Programming Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
Machine Learning Model: Decision Tree
Deployment: Streamlit Web App
📑 Project Workflow (ML Pipeline)
🔹 1️⃣ Data Loading
📌 Load the dataset using Pandas and explore basic statistics.

python
Copy
Edit
import pandas as pd
df = pd.read_csv("hospital_data.csv")
df.head()
🔹 2️⃣ Data Preprocessing & Cleaning
📌 Handle missing values, duplicate records, and incorrect data entries.

Convert categorical columns into numerical values using Label Encoding
Apply feature scaling to numerical columns using StandardScaler
python
Copy
Edit
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encoding categorical features
label_encoders = {}
categorical_cols = ["Hospital_type_code", "Department", "Ward_Type", "Type of Admission", "Severity of Illness"]
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Scaling numerical features
scaler = StandardScaler()
df[['Type of Admission', 'Admission_Deposit']] = scaler.fit_transform(df[['Type of Admission', 'Admission_Deposit']])
🔹 3️⃣ Exploratory Data Analysis (EDA)
📌 Perform data visualization using Matplotlib and Seaborn.

python
Copy
Edit
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing Length of Stay distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=df["Stay"])
plt.title("Distribution of Patient Length of Stay")
plt.show()
🔹 4️⃣ Model Building
📌 Train a Decision Tree model to predict LOS.

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Splitting data
X = df.drop(columns=["Stay"])
y = df["Stay"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
import pickle
with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)
🔹 5️⃣ Model Evaluation
📌 Check accuracy and confusion matrix.

python
Copy
Edit
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
🔹 6️⃣ Deploying the Model with Streamlit
📌 Create a user-friendly web app for predictions.

python
Copy
Edit
import streamlit as st

# Load trained model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Patient Length of Stay Prediction")

# Take user input
hospital_type = st.selectbox("Hospital Type Code:", [0, 1, 2, 3, 4, 5, 6])
department = st.selectbox("Department:", [0, 1, 2, 3, 4])
ward_type = st.selectbox("Ward Type:", [0, 1, 2, 3, 4, 5])
severity = st.selectbox("Severity of Illness:", [0, 1, 2])
admission_deposit = st.number_input("Admission Deposit:")

# Create input DataFrame
input_data = pd.DataFrame([[hospital_type, department, ward_type, severity, admission_deposit]])
input_data.columns = ["Hospital_type_code", "Department", "Ward_Type", "Severity of Illness", "Admission_Deposit"]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Length of Stay: {prediction}")
🚀 Steps to Upload Project on GitHub
🔹 1️⃣ Create a GitHub Repository
Go to GitHub
Click New Repository → Enter Project Name
Choose Public or Private
Click Create Repository
🔹 2️⃣ Initialize Git in Your Local Project
Open Terminal / Command Prompt in your project folder:

bash
Copy
Edit
git init
git add .
git commit -m "Initial commit - Patient LOS Prediction"
🔹 3️⃣ Connect to GitHub & Push Code
Copy the repository URL from GitHub and run:

bash
Copy
Edit
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
🔹 4️⃣ Clone Repository (if needed in future)
bash
Copy
Edit
git clone <your-repo-url>
📌 Conclusion
Developed an ML model for hospital Length of Stay prediction
Used feature engineering, visualization, and model evaluation
Deployed it as a Streamlit web app
Uploaded code to GitHub for future collaboration
This project demonstrates the power of data science in healthcare management, optimizing hospital efficiency and patient care! 
