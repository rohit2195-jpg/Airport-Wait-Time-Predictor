# ✈️ Airport Wait Time Predictor

##  Overview

This project leverages machine learning to predict airport wait times. Each airport had a seperate model to make it easier to train.

After testing multiple machine learning algorithms, the best-performing model was **XGBoost**, a gradient boosting technique. Other models such as **Linear Regression**, **K-Nearest Neighbors (KNN)**, and **Neural Networks** were also implemented, but did not perform as well in terms of accuracy.



The Mean Absolute Error and r2 values of these models can be further improved with more hymperparameter optimization and better results can be yielded. 



## 📊 Model Performance by Airport

| Airport | Mean Squared Error (MSE) | Mean Absolute Error (MAE) | R² Score |
|---------|---------------------------|----------------------------|----------|
| MIA     | 106.81                    | 7.54                       | 0.217    |
| BNA     | 84.41                     | 6.85                       | 0.441    |
| ATL     | 53.11                     | 5.05                       | 0.349    |
| SLC     | 24.29                     | 3.72                       | 0.308    |
| SMF     | 31.09                     | 3.80                       | 0.219    |
| MSP     | 99.37                     | 7.07                       | 0.330    |
| SAN     | 46.58                     | 4.96                       | 0.225    |
| CVG     | 22.13                     | 3.78                       | 0.166    |
| IAH     | 68.72                     | 6.22                       | 0.462    |
| HNL     | 70.05                     | 6.02                       | 0.400    |
| PHX     | 27.16                     | 3.46                       | 0.191    |
| PDX     | 16.21                     | 3.10                       | 0.276    |
| CLT     | 75.59                     | 5.99                       | 0.406    |
| PHL     | 48.12                     | 4.34                       | 0.214    |
| SPN     | 56.02                     | 5.21                       | 0.354    |
| LAX     | 149.78                    | 9.34                       | 0.239    |
| RDU     | 57.98                     | 5.58                       | 0.112    |
| SEA     | 76.28                     | 6.65                       | 0.421    |
| BOS     | 51.32                     | 5.42                       | 0.346    |
| PVD     | 6.58                      | 2.01                       | -12.930  |
| ORD     | 106.59                    | 7.61                       | 0.555    |
| TPA     | 56.56                     | 5.51                       | 0.284    |
| BWI     | 36.42                     | 4.59                       | 0.269    |
| MCO     | 211.33                    | 11.40                      | 0.068    |
| SAT     | 41.38                     | 4.47                       | 0.177    |
| DEN     | 20.19                     | 3.25                       | 0.371    |
| FLL     | 140.77                    | 8.73                       | 0.202    |
| DFW     | 41.17                     | 4.64                       | 0.435    |
| IAD     | 98.33                     | 7.35                       | 0.270    |
| SFO     | 84.54                     | 6.91                       | 0.308    |

## Model Performance & Real-World Context

This project aimed to predict airport wait times using machine learning models. Results varied across airports, with R² scores ranging from **0.55** (moderate accuracy) to negative values (indicating the model underperformed).

### Why Some Models Performed Well
- **Predictable patterns**: Airports like ORD and IAH had more consistent wait time trends.
- **Better data quality**: Clean, structured data helped the model learn meaningful patterns.

### Why Some Models Struggled
- **High variability**: Airports like MCO and PVD had unpredictable fluctuations in wait times.
- **Data limitations**: Incomplete, noisy, or inconsistent data affected model learning.
- **Negative R²**: Indicates the model performed worse than a baseline, included here for transparency.

### Why This Still Matters
Even with modest R² scores (0.3–0.5), these models show potential in identifying trends. In a real-world setting, this can still offer **operational value**. This project lays the groundwork for future improvements, such as incorporating real-time data or airport-specific modeling strategies.



## Tech Stack

- **Frontend**: HTML, JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: XGBoost, scikit-learn, pandas, NumPy

Trained models are saved in the `saved_models/` directory.

---

##  Testing Instructions

1. Start the backend:
   ```bash
   cd backend
   python backend.py

2. Start the frontend by running index.html

# Useful Links
1. Link to dataset {https://awt.cbp.gov}

##  Future Work

- Add support for additional airports
- Improve model accuracy
- Deploy app on AWS
