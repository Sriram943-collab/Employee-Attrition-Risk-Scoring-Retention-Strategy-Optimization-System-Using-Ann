# 🚀 AI Employee Attrition Risk Scoring & Retention Strategy Optimization System

An advanced **AI-powered decision intelligence system** that predicts employee attrition risk using a **Deep Learning (Artificial Neural Network - ANN)** model and generates **data-driven retention strategies** through intelligent analysis.

---

## 📌 Problem Statement

Employee attrition is a critical issue faced by organizations, leading to:

- Increased recruitment and training costs  
- Loss of experienced workforce  
- Decreased productivity and morale  

This project addresses a **supervised binary classification problem**, where the objective is to predict whether an employee is likely to:

- **Stay (0)**  
- **Leave (1)**  

The system extends beyond prediction to provide **actionable business insights**, making it a complete **decision-support system**.

---

## 🧠 Solution Approach

This project transforms a traditional DL pipeline into a **multi-layer AI system**:

1. Predict attrition probability using ANN  
2. Convert prediction into an interpretable **risk score**  
3. Identify **key contributing factors**  
4. Generate **optimized retention strategies**  
5. Deliver insights through an interactive dashboard  

---

## 🏗️ System Architecture
- Employee Data
↓
- Data Preprocessing (Cleaning + Encoding + Scaling + Feature Engineering)
↓
- Artificial Neural Network (ANN Model)
↓
- Probability Output (0–1)
↓
- Risk Scoring Engine (0–100%)
↓
- Root Cause Analysis Engine
↓
- Retention Strategy Optimization
↓
- Interactive Dashboard (Streamlit)

---

## ⚙️ Data Preprocessing

- Handling missing values and duplicates  
- Encoding categorical variables (One-Hot Encoding)  
- Feature engineering:
  - Income transformations  
  - Promotion rate  
  - Work stress index  
- Feature scaling using StandardScaler  

---

## 🤖 Model Architecture (ANN)

The model is built using **Keras Sequential API** with optimized architecture:

- **Input Layer**
  - Number of neurons = number of features  

- **Hidden Layers**
  - Multiple Dense layers (tuned using Optuna)
  - Activation Functions:
    - ReLU / Tanh
  - Regularization:
    - Dropout (to prevent overfitting)
    - L1 Regularization
  - Batch Normalization (for stable learning)

- **Output Layer**
  - 1 neuron
  - Activation: Sigmoid (for binary classification)

---

## ⚡ Model Training

- **Loss Function:** Binary Crossentropy  
- **Optimizers Used:** Adam / RMSprop / SGD (tuned)  
- **Evaluation Metrics:**
  - Accuracy  
  - Precision  
  - Recall  
- **Early Stopping:** Prevents overfitting  
- **Hyperparameter Tuning:** Optuna optimization  

---

## 📊 Risk Scoring System

Instead of raw predictions, the model output is transformed into:

- **Risk Score (0–100%)**
- **Risk Levels:**
  - Low  
  - Moderate  
  - High  
  - Critical  

This makes the model output **interpretable for business users**.

---

## 🔍 Root Cause Analysis

The system evaluates employee attributes to identify **key drivers of attrition**, such as:

- Low compensation  
- Poor work-life balance  
- Lack of promotions  
- Low job satisfaction  
- Long commute distance  

Each factor is assigned an **impact score** for prioritization.

---

## 🎯 Retention Strategy Optimization

A rule-based intelligent engine generates **targeted retention strategies**:

- 💰 Salary Revision  
- 📈 Career Growth Planning  
- ⚖️ Workload Optimization  
- 🧠 HR / Manager Intervention  
- 🏠 Flexible Work Options  

Strategies are dynamically adjusted based on **risk level**:

- Low → No action  
- Moderate → Preventive action  
- High/Critical → Immediate intervention  

---

## 📊 Output Dashboard

The system provides a structured output:

- 🔥 Risk Score  
- ⚠️ Risk Level  
- 📊 Confidence Level  
- 🔍 Root Cause Analysis  
- 🎯 Recommended Retention Strategies  
- 📄 Downloadable Report  

---

## 💡 Business Impact

This system enables organizations to:

- Proactively identify at-risk employees  
- Reduce attrition-related costs  
- Improve employee satisfaction and engagement  
- Enhance HR decision-making  
- Build data-driven retention strategies  

---

## 🧠 Key Innovations

- Transformation of ML model → **AI decision intelligence system**  
- Integration of prediction with **business logic**  
- Focus on **actionable insights instead of raw accuracy**  
- End-to-end pipeline from data → decision  

---

## 🏆 Conclusion

This project demonstrates how **deep learning and business intelligence** can be combined to solve real-world problems. It goes beyond prediction to provide a **complete AI-driven decision system** for employee retention.

---

## 👨‍💻 Author

**Krishnasagarapu Sri Ram**

---
