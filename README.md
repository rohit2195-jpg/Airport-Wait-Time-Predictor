# ✈️ Airport Wait Time Predictor

##  Overview

This project leverages machine learning to predict airport wait times. Each airport had a seperate model to make it easier to train.

After testing multiple machine learning algorithms, the best-performing model was **XGBoost**, a gradient boosting technique. Other models such as **Linear Regression**, **K-Nearest Neighbors (KNN)**, and **Neural Networks** were also implemented, but did not perform as well in terms of accuracy.

Overall, some models better fit the airports than others, and better predicted the wait times as shown below. Mean Absolute Error was chosen as the metric because it better reflected real-world performance and provided intuitive interpretability for practical applications. However, certain models had high r2 values too. Ex. ORD - 0.556 and DFW - 0.434 and SEA - 0.421.  

I think the Mean Absolute Error and r2 values of these models can be further improved with more hymperparameter optimization. When analyzing the results, it became clear that the model had a harder time accurately predicting wait times at larger airports with multiple terminals. This makes intuitive sense because in such complex environments, wait times can fluctuate quickly and unevenly across different terminals. Factors like varying passenger volumes, security procedures, and staffing changes contribute to this variability, making it more challenging for the model to capture all the nuances and produce precise predictions.

## Model Performance

| Airport Code | Mean Absolute Error |
|--------------|---------------------|
| MIA          | 7.5395              |
| BNA          | 6.8510              |
| ATL          | 5.0450              |
| SLC          | 3.7238              |
| SMF          | 3.7983              |
| MSP          | 7.0690              |
| SAN          | 4.9620              |
| CVG          | 3.7771              |
| IAH          | 6.2210              |
| HNL          | 6.0193              |
| PHX          | 3.4564              |
| PDX          | 3.0974              |
| CLT          | 5.9932              |
| PHL          | 4.3394              |
| SPN          | 5.2133              |
| LAX          | 9.3391              |
| RDU          | 5.5837              |
| SEA          | 6.6512              |
| BOS          | 5.4205              |
| PVD          | 2.0114              |
| ORD          | 7.6055              |
| TPA          | 5.5105              |
| BWI          | 4.5937              |
| MCO          | 11.3997             |
| SAT          | 4.4692              |
| DEN          | 3.2493              |
| FLL          | 8.7291              |
| DFW          | 4.6370              |
| IAD          | 7.3506              |
| SFO          | 6.9149              |



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
