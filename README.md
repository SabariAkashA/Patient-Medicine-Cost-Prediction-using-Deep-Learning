## 🧠 Patient Medicine Cost Prediction using Deep Learning

A project focused on predicting a patient’s total medicine cost using demographic and clinical data, powered by a **Feedforward Neural Network (FNN)** and visualized through a **Streamlit dashboard**.

---
### 📌 Problem Statement

Hospitals and insurance providers need to anticipate patient treatment costs for budgeting and policy decisions. This project uses deep learning to predict **medicine cost** based on patient data such as age, gender, diagnosis, medication, insurance, and length of stay.

---

### ⚙️ Tools & Technologies

* **Python**
* **Pandas, NumPy** – data manipulation
* **Scikit-learn** – preprocessing
* **TensorFlow / Keras** – FNN model
* **Streamlit** – interactive dashboard
* **Matplotlib / Seaborn** – visualizations
---

### 🛠️ Workflow

1. **Data Cleaning** – Removed duplicates, handled missing values
2. **Feature Engineering** – Added `risk_score`, `age_group`, `length of stay`
3. **Preprocessing** – One-hot encoding, normalization
4. **Modeling** – Feedforward Neural Network with multiple layers
5. **Evaluation** – MAE, MSE, and R² metrics
6. **Deployment** – Streamlit dashboard to take input and show cost predictions

---

### 📊 Model Performance

| Metric       | Value  |
| ------------ | ------ |
| **MAE**      | ₹12285.11 |
| **MSE**      | ₹202048037.1 |
| **R² Score** | -0.0028   |

---

### 🖼 Dashboard Preview

![Dashboard](https://github.com/user-attachments/assets/e6a38250-678e-4903-9838-dc5a0de78b0c)

---

### 🚀 How to Run

1. Clone the repo
2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```
3. Launch dashboard:
   ```bash
   !npm install localtunnel
   !streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py &>/dev/null&
   ```
   
   ```bash
   streamlit run app.py
   ```
   ```bash
   !pip install pyngrok
   from pyngrok import ngrok
   ngrok.set_auth_token("2vtcH3d1WIAyMdu7psEn0fdJto0_QZMtf5jLBQqsdpyQa1m6")  # Replace with your ngrok token
   !streamlit run app.py & npx localtunnel --port 8501
   ```
---

### 📌 Sample Prediction

> Predicted billing cost for a 26-year-old male cancer patient with a 7-day stay:
> **₹12,622.54**

![image](https://github.com/user-attachments/assets/8aad0b36-552f-4787-bcda-6b9c328ae402)

---


