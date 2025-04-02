# Bank Marketing Campaign Prediction

This project is focused on predicting whether a client will subscribe to a bank deposit based on various demographic and financial features. The goal is to build and evaluate machine learning models to help banks optimize their marketing campaigns by targeting customers with the highest likelihood of opening a deposit.

## Project Overview

In this project, we apply a variety of classification algorithms to predict the success of a bank's marketing campaign. We use a dataset provided by the bank that includes customer details such as age, job, marital status, and balance, among others.

The models implemented in this project include:
- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting**

These models are trained and evaluated to predict whether a customer subscribes to a deposit or not. We aim to find the most effective model based on accuracy, precision, recall, and F1 score.

## Dataset

The dataset used in this project is **bank_fin.csv**, which contains the following features:

- **age**: Age of the client
- **job**: Type of job
- **marital**: Marital status
- **education**: Education level
- **default**: Whether the client has credit in default
- **balance**: Balance of the client
- **housing**: Whether the client has a housing loan
- **loan**: Whether the client has a personal loan
- **contact**: Contact communication type
- **month**: Last contact month of year
- **duration**: Duration of the last contact in seconds
- **campaign**: Number of contacts performed during this campaign
- **pdays**: Number of days since the client was last contacted
- **previous**: Number of contacts performed before this campaign
- **poutcome**: Outcome of the previous marketing campaign
- **deposit**: Target variable (whether the client subscribed to a deposit)

### Example Rows

| age | job         | marital | education | balance | housing | loan | contact | month | duration | ... |
|-----|-------------|---------|-----------|---------|---------|------|---------|-------|----------|-----|
| 59  | admin.      | married | secondary  | 2343    | yes     | no   | cellular| may   | 1042     | ... |
| 56  | technician  | married | secondary  | 45      | no      | no   | cellular| may   | 1467     | ... |
| 41  | services    | married | secondary  | 1270    | yes     | no   | cellular| may   | 1389     | ... |

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Alex1988Den/DS-Project-4-ML-Classification.git
   cd DS-Project-4-ML-Classification

    Install necessary dependencies: You can install the required Python packages by running:

pip install -r requirements.txt

Run the Jupyter notebook: Open the notebook file Project_4_ML.ipynb using Jupyter:

    jupyter notebook Project_4_ML.ipynb

    The notebook contains all the necessary code for data preprocessing, model training, evaluation, and performance comparison.

Models and Evaluation

The following machine learning models were implemented and evaluated in this project:

    Logistic Regression: A simple, interpretable linear model.

    Decision Trees: A model that splits the data based on decision rules.

    Random Forest: An ensemble of decision trees that reduces overfitting.

    Gradient Boosting: A model that builds trees sequentially, correcting previous errors.

Performance Metrics:

The models were evaluated using the following metrics:

    Accuracy: Overall percentage of correct predictions.

    Precision: Proportion of positive predictions that are actually correct.

    Recall: Proportion of actual positives that were correctly identified.

    F1 Score: Harmonic mean of precision and recall.

    ROC AUC: Performance measure for classification problems, particularly when class imbalance is present.

Results:

    Best Model: Random Forest and Gradient Boosting achieved the best performance, with a high ROC AUC score.

Example of Results:

    Logistic Regression:

        Accuracy: 81.5%

        Precision: 82%

        Recall: 76.9%

        F1 Score: 79.4%

        ROC AUC: 89.6%

    Random Forest:

        Accuracy: 84.7%

        F1 Score: 84%

        ROC AUC: 91.4%

Future Improvements

    Hyperparameter Tuning: Implementing GridSearchCV or Optuna for hyperparameter optimization.

    Feature Engineering: Adding more features or transforming existing features might improve model performance.

    Deep Learning: Experimenting with deep learning models like Neural Networks for potentially better results.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    Bank Marketing Dataset from UCI Machine Learning Repository.

    Thanks to all contributors and libraries used, including Scikit-learn, Pandas, NumPy, and Matplotlib.
