# 🏡 Airbnb Income Classification with Logistic Regression   
BTTAI Machine Learning Foundations Lab 5

## About The Project  
This project implements a complete machine learning pipeline to predict a binary outcome from Airbnb listing data using logistic regression. The goal is to explore model performance through baseline training, hyperparameter tuning, and feature selection using the `scikit-learn` library.

---

### 📌 Project Objectives

- ✅ Load and explore the Airbnb "listings" dataset
- ✅ Define the prediction task: **binary classification** based on listing attributes
- ✅ Identify and select relevant features and label
- ✅ Create labeled examples and split the dataset into **training** and **test** sets
- ✅ Train and evaluate a **baseline logistic regression model** using default parameters
- ✅ Perform **grid search** to optimize the regularization parameter `C`
- ✅ Train and evaluate the **tuned model** using the optimal hyperparameter
- ✅ Plot and compare **Precision-Recall curves** for both models
- ✅ Plot and compare **ROC curves** and compute **AUC scores**
- ✅ Apply **feature selection** techniques to refine model inputs
- ✅ **Persist the final model** to disk for future inference

---

### 🧰 Tools & Libraries

- Python 3.x
- `pandas`, `numpy` for data manipulation
- `scikit-learn` for model training, evaluation, and feature selection
- `matplotlib`, `seaborn` for visualization
- `pickle` for model persistence

---

### 📈 Example Outputs

- Baseline model accuracy, AUC, and log loss
- Optimized logistic regression performance after tuning
- Visual comparisons of ROC and Precision-Recall curves
- Ranked list of top predictive features

---

### 🚀 Getting Started

To reproduce this project:

1. Clone the repo
2. Install dependencies  
