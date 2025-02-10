# Length-of-stay-in-hospital-ML-Project
Part of college project

****************************************************

# 🛏️ Patient Length of Stay Prediction  

## 📌 Project Overview  
The healthcare sector has always embraced technological advancements to enhance patient care and hospital management. With the rise of **machine learning**, hospitals can now leverage data to optimize **staff allocation, resource management, and patient treatment plans**.  

This project focuses on predicting **patient length of stay (LOS)** at the time of admission using **machine learning techniques**. Predicting LOS helps in **effective hospital resource allocation**, reducing patient overload, and **minimizing the risk of infections**.  

## 🩺 Problem Statement  
The **COVID-19 pandemic** highlighted the importance of **efficient healthcare management**, particularly in managing **hospital resources and patient flow**. One **critical metric** for hospital efficiency is **patient Length of Stay (LOS)**—the duration a patient stays in the hospital.  

### **Objective**  
To **predict the LOS** of a patient using **historical hospital data**, allowing hospitals to:  
✅ Identify high-risk patients who may require prolonged care.  
✅ Optimize treatment plans to reduce LOS.  
✅ Improve logistics (bed allocation, staffing, equipment management).  

### **LOS Categories (Target Variable)**  
The LOS is classified into **11 different categories**, ranging from **0-10 days to more than 100 days**.  

---

## 📂 Dataset Overview  
The dataset contains multiple features related to hospital admissions, such as:  
- **Patient demographics** (Age, Admission type)  
- **Hospital details** (Department, Ward Type, Facility Code)  
- **Medical condition severity** (Severity of Illness)  
- **Financials** (Admission Deposit)  

---

## 🛠️ Tech Stack Used  
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit  
- **Machine Learning Model**: Decision Tree  
- **Deployment**: Streamlit Web App  

---

## 📑 Project Workflow (ML Pipeline)  

### 🔹 1️⃣ Data Loading  
- Load the dataset using Pandas.  

### 🔹 2️⃣ Data Preprocessing & Cleaning  
- Handle missing values, duplicate records, and incorrect data entries.  
- Convert categorical columns into numerical values.  
- Apply feature scaling to numerical columns.  

### 🔹 3️⃣ Exploratory Data Analysis (EDA)  
- Perform data visualization using Matplotlib and Seaborn.  
- Analyze feature correlations and distributions.  

### 🔹 4️⃣ Model Building  
- Train a **Decision Tree Classifier** model to predict LOS.  

### 🔹 5️⃣ Model Evaluation  
- Check accuracy and confusion matrix.  

### 🔹 6️⃣ Deployment using Streamlit  
- Create a **user-friendly web app** for making predictions.  

---

## 🚀 How to Run the Project  

### 1️⃣ Clone the Repository  
```bash
git clone <your-repo-url>
cd patient-los-prediction
```
### 2️⃣ Install Required Libraries  
Run the following command to install all necessary dependencies:  
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit Web App
Execute the following command to launch the web application:
```bash
streamlit run app.py
```



📌 Steps to Upload Project on GitHub
1️⃣ Create a GitHub Repository
Go to GitHub
Click New Repository → Enter Project Name
Choose Public or Private
Click Create Repository
2️⃣ Initialize Git in Your Local Project
```
git init
git add .
git commit -m "Initial commit - Patient LOS Prediction"
```
3️⃣ Connect to GitHub & Push Code
```
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

📌 Conclusion
Successfully developed and deployed a machine learning model to predict patient Length of Stay (LOS) in hospitals.
Applied data preprocessing, feature engineering, visualization, and model evaluation techniques.
Deployed the model using Streamlit, providing an interactive web interface.
Uploaded the project to GitHub, ensuring collaboration and future improvements.
This project highlights how machine learning can optimize hospital efficiency and patient management, contributing to better healthcare outcomes! 
