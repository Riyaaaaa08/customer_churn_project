# 📊 Telco Customer Churn Predictor

An end-to-end Machine Learning pipeline built to identify at-risk telecom customers and combat business churn. This project focuses on handling class imbalances to optimize real-world business metrics like **Recall**.

## 🚀 Project Overview & Business Impact

In the telecom industry, acquiring a new customer costs significantly more than retaining an existing one. This project analyzes a dataset of 7,043 customers to predict who is likely to leave.

By identifying these accounts before they drop their service, marketing and retention teams can step in with targeted incentives to secure long-term loyalty.

## 🔧 Core Pipeline Architecture

The codebase is split into modular, production-ready components:

- `data_preprocessing.py`: Features a defensive data engineering pipeline that cleans anomalies, handles hidden missing values in `TotalCharges`, executes One-Hot Encoding to eliminate the Dummy Variable Trap, and scales features via `StandardScaler`.
- `train.py`: Handles model initialization, evaluation comparisons, and class-balancing configurations.

## 🎯 The Machine Learning Journey (Results)

The primary business objective was maximizing **Recall** (minimizing the number of churning customers our system misses completely).

| Model Config                     | Total Accuracy | Churn Precision | Churn Recall (Catch Rate) |
| :------------------------------- | :------------: | :-------------: | :-----------------------: |
| Baseline Logistic Regression     |      81%       |      0.66       |            57%            |
| Random Forest Classifier         |      79%       |      0.62       |            49%            |
| **Balanced Logistic Regression** |    **74%**     |    **0.51**     |          **78%**          |

### 🧠 Strategic Technical Trade-off

While the baseline model had a higher raw accuracy (81%), it failed to catch 43% of leaving customers. By implementing `class_weight='balanced'`, we shifted the model's decision boundary. This successfully traded a minor amount of precision to **boost our churn catch rate to 78%**, directly protecting company revenue.

## 🛠️ Setup Instructions

To run this pipeline locally on your machine:

1. Clone this repository:
   ```bash
   git clone [https://github.com/Riyaaaaa08/customer_churn_project.git](https://github.com/Riyaaaaa08/customer_churn_project.git)
   cd customer_churn_project
   ```
