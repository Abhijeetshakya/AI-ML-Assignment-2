# Customer Churn Prediction using Logistic Regression

**Name:** Abhijeet Shakya
**Registration No.:** 23BCE10383
**Application No.:** IN26010262

## Objective
A telecommunications company wants to predict whether a customer is likely to **churn** (leave the service) based on their demographic information and service usage patterns. This project builds and evaluates a **Logistic Regression** model to make that prediction, so the company can proactively target at-risk customers with retention offers.

## Dataset
**Telco Customer Churn** — 7,043 customers, 21 features (demographics, account info, and subscribed services), with `Churn` (Yes/No) as the target variable.

- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

> The dataset is **not included in this repository**. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the Kaggle link above, rename it to `telco.csv`, and place it in the project root before running the notebook/script.

## Libraries Used
- `pandas`, `numpy` — data loading and manipulation
- `scikit-learn` — preprocessing, train/test split, Logistic Regression, evaluation metrics
- `matplotlib`, `seaborn` — visualization (confusion matrix heatmap)

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, and identified numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`), categorical features (16 columns such as `gender`, `Contract`, `InternetService`, `PaymentMethod`, etc.), and the target variable (`Churn`).
2. **Data Preprocessing**
   - The `customerID` column was dropped (a unique identifier with no predictive value).
   - `TotalCharges` was coerced to numeric (it contained blank strings for 11 new customers), and the 11 rows with missing `TotalCharges` were dropped.
   - The target `Churn` was label-encoded to 0/1.
   - Categorical features were one-hot encoded (`pd.get_dummies`, drop-first to avoid multicollinearity).
   - Numerical features were standardized with `StandardScaler`.
   - The data was split **80% train / 20% test** (`random_state=42`).
3. **Model Development** — Trained a `LogisticRegression` model (scikit-learn, `max_iter=1000`) on the training set and generated predictions on the held-out test set.
4. **Model Evaluation** — Evaluated the model using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

#RESULT
Accuracy Score: 0.7934
Precision: 0.6424
Recall: 0.5027
F1-Score: 0.5640

Confusion Matrix:
[[1392  157]
 [ 279  282]]
![Confusion Matrix](confusion_matrix.png)

**Top influential features** (by absolute Logistic Regression coefficient): `tenure` (−), `TotalCharges` (+), `InternetService_Fiber optic` (+), `MonthlyCharges` (−), `Contract_Two year` (−), `Contract_One year` (−).

### Observations
1. The model reaches **~79% overall accuracy**, but recall on the churn class (0.52) is noticeably lower than on the no-churn class (0.89) — a direct effect of class imbalance (~73% No-Churn vs. ~27% Churn in the raw data).
2. **Tenure** and **longer contract lengths** are the strongest protective factors against churn, while **fiber-optic internet**, **TotalCharges**, and **higher monthly charges** are the strongest risk factors — pointing to price-sensitive, newer customers as the highest-risk segment.
3. Precision on the churn class (0.62) means a meaningful fraction of customers flagged as "will churn" don't actually churn, which matters if flagged customers receive costly retention incentives.

## Conclusion
This project used a Logistic Regression model to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. Rows with missing `TotalCharges` values were dropped and categorical variables were one-hot encoded before an 80/20 train-test split. The model achieved an accuracy of about 79%, with a precision of 0.62 and recall of 0.52 for the churn class. The results show that **contract type**, **tenure**, and **monthly/total charges** are the strongest predictors of churn: customers on month-to-month contracts with short tenure and high charges (often for fiber-optic internet) are most likely to leave, while long-tenured customers on annual contracts are the most loyal. These findings suggest that retention efforts should focus on new, high-paying, month-to-month customers, for example through loyalty incentives or contract upgrades. A key limitation of Logistic Regression here is that it assumes a **linear relationship** between the features and the log-odds of churn, and cannot naturally capture complex, non-linear interactions between features (e.g., how contract type and internet service jointly affect churn) the way tree-based models like Random Forest or Gradient Boosting can.


