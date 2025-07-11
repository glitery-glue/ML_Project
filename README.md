
```markdown
#  Student Performance Prediction using Machine Learning

This project is an **End-to-End Machine Learning pipeline** designed to analyze and predict student performance based on various socio-demographic and educational factors. The goal is to build and evaluate regression models that can predict scores in **math, reading, or writing**.

---



##  Project Overview

This machine learning pipeline focuses on:
- Data ingestion
- Data preprocessing and transformation
- Model training with hyperparameter tuning
- Model evaluation and selection
- Prediction using the best-trained model
- Interactive prediction via a **Streamlit web app**

Target variable: `math_score`, `reading_score`, or `writing_score`

---



---

## Features

 Data ingestion from CSV  
 Data transformation (missing value imputation, scaling, encoding)  
 Multiple regression models (Random Forest, XGBoost, CatBoost, etc.)  
 Hyperparameter tuning using GridSearchCV  
 Model evaluation using R² Score  
 Best model tracking and saving  
 Streamlit web interface for live predictions  
 Modular code structure for reusability and clarity  

---



##  Workflow Pipeline

1. **Data Ingestion**  
   Loads raw dataset and splits it into train/test.

2. **Data Transformation**  
   Handles missing values, encodes categorical features, and scales numerical ones.

3. **Model Training**  
   - Trains multiple models
   - Performs hyperparameter tuning
   - Selects best-performing model

4. **Model Evaluation**  
   Saves model performance metrics and hyperparameters.

5. **Prediction**  
   Uses the saved best model to predict on new data.

---

##  Model Evaluation

- Metric: **R² Score**
- Models used:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - CatBoost
  - AdaBoost
  - Decision Tree

The best-performing model and its parameters are saved for inference.

---




## Future Improvements

* Add explainability (e.g., SHAP values)
* Model versioning and deployment with MLflow
* Store results in a database
* Add CI/CD pipeline for automation
* Dockerize the application

---

## Acknowledgements

This project is inspired by practical implementations of real-world ML systems and serves as a portfolio-ready example of MLOps in action.

---

## Author

**Sudeshna Saha**

```
