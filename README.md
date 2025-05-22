# ✈️ Airport Wait Time Predictor for Chicago O'Hare Airport

## 📌 Overview

This project leverages machine learning to predict airport wait times. Currently, the model supports **Chicago O'Hare International Airport**, with plans to expand to additional airports in the future.

After testing multiple machine learning algorithms, the best-performing model was **XGBoost**, a gradient boosting technique. Other models such as **Linear Regression**, **K-Nearest Neighbors (KNN)**, and **Neural Networks** were also implemented, but did not perform as well in terms of accuracy.

## Model Performance

| Metric              | Value         |
|---------------------|---------------|
| Mean Absolute Error | 7.5 minutes   |
| R² Score            | 0.555         |
| Model Used          | XGBoost       |


## Tech Stack

- **Frontend**: HTML, JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: XGBoost, scikit-learn, pandas, NumPy

Trained models are saved in the `saved_models/` directory.

---

## 🧪 Testing Instructions

1. Start the backend:
   ```bash
   cd backend
   python backend.py

2. Start the frontend by running index.html

# Useful Links
1. Link to dataset {https://awt.cbp.gov}

## 🚀 Future Work

- Add support for additional airports
- Improve model accuracy
- Deploy app on AWS
