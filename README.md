# Fraud Detection API

Machine Learning API for detecting fraudulent credit card transactions using XGBoost and FastAPI.

---

## 🚀 Features

- Fraud detection using Machine Learning
- XGBoost classification model
- REST API built with FastAPI
- Real-time prediction endpoint
- Interactive Swagger API documentation
- Model persistence using Joblib

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Joblib

---

## 📂 Project Structure

```bash
fraud-detection-api/
│
├── data/
├── models/
├── src/
├── notebooks/
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚡ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run API

```bash
uvicorn main:app --reload
```

---

## 📖 API Documentation

After starting server:

```bash
http://127.0.0.1:8000/docs
```

---

## 📊 Model Information

- Algorithm: XGBoost Classifier
- Task: Binary Classification
- Dataset: Credit Card Fraud Detection Dataset
- Output:
  - 0 = Normal Transaction
  - 1 = Fraudulent Transaction

---

## 🧠 Future Improvements

- Docker containerization
- Cloud deployment
- Frontend dashboard
- Authentication
- Database integration