# Airbnb Paris Price Prediction

Predicting prices of Airbnb listings in Paris using open datasets and regression models.  

**Why it matters:**  
Accurate pricing reduces friction in two-sided marketplaces, boosting user trust and transaction volume.  
In fintech, this mirrors tasks like predicting FX spreads, loan rates, or transaction fees, exactly the type of data-driven optimisation used to improve products, increase retention, and drive growth.

---

## Project Overview

This project follows a complete data science workflow:
1. **EDA** – explore distributions, detect anomalies, assess correlations.
2. **Feature Engineering** – build geospatial, temporal, and demand-related features.
3. **Model Training & Comparison** – benchmark algorithms for predictive performance.
4. **Evaluation & Interpretation** – measure accuracy, interpret drivers, and present actionable insights.

---

## Tech Stack

- **Python**: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`
- **Jupyter Notebooks** for exploration
- **Git/GitHub** for version control

---

## Repo Structure
.
├─ notebooks/ 
  └─ reports/figures/
│     ├─ correlation_matrix.png
│     ├─ log_price_by_room_type.png
│     ├─ model_comparison.png
│     ├─ price_distribution_log.png
│     └─ top_neighborhoods.png
│ └─ 01_eda.ipynb
│ └─ 02_feature_engineering.ipynb
│ └─ 03_model_comparison.ipynb
├─ src/
│  ├─ __pycache__/
│  ├─ data_loader.py
│  ├─ evaluate_model.py
│  ├─ features.py
│  └─ train_model.py
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt


## Features Used

- **Property characteristics**: property type, room type  
- **Location**: neighbourhood, coordinates, distance to centre  
- **Demand signals**: availability, number of reviews  
- **Host metrics**: superhost status, number of listings  
- **Text/amenity features**: description length, amenities count

---

## Key Visuals

### Correlation Matrix
Shows relationships between variables and identifies strong predictors of price.  
![Correlation Matrix](reports/figures/correlation_matrix.png)

### Top 15 Neighborhoods by Listing Count
Highlights supply distribution in Paris.  
![Top Neighborhoods](reports/figures/top_neighborhoods.png)

### Price by Room Type
Differences in log-transformed prices across room categories.  
![Log Price by Room Type](reports/figures/log_price_by_room_type.png)

### Log-transformed Price Distribution
Normalised distribution for modelling stability.  
![Price Distribution](reports/figures/price_distribution_log.png)

---

## Models Compared

- **XGBoost**  
- **Random Forest**  
- **Ridge Regression**

---

## Evaluation Metrics

- **MAE** – Mean Absolute Error (average absolute deviation from true price)  
- **RMSE** – Root Mean Squared Error (penalises large deviations)  
- **R²** – Variance explained by the model  
---

## Model Performance

| Model         | MAE (log) | RMSE (log) | R²     |
|---------------|-----------|------------|--------|
| XGBoost       | 0.390     | 0.499      | 0.4427 |
| Random Forest | 0.392     | 0.503      | 0.4327 |
| Ridge         | 0.482     | 0.607      | 0.1747 |

**Performance Plot:**  
![Model Comparison](reports/figures/model_comparison.png)

**Interpretation:**  
- **XGBoost** had the lowest error and highest R².
- Tree-based models outperformed linear methods, capturing non-linear location and amenity effects.

---

## License & Attribution

- MIT License
- Data © [InsideAirbnb](http://insideairbnb.com/get-the-data.html)
