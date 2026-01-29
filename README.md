# Automated-Credit-Card-Approval-Prediction

## Problem Statement

Commercial banks receive a large volume of credit card applications every day. Manually reviewing these applications is time-consuming, error-prone, and costly.

The goal of this project is to automate the credit card approval process by training a machine learning model that predicts whether an application should be approved or rejected based on applicant information.

## Dataset

Source: UCI Machine Learning Repository (Credit Card Approval dataset)

Each row represents a credit card application

The dataset contains:

Numerical features (e.g. financial indicators)

Categorical features (e.g. demographic attributes)

The target variable is the final column:

+ → Approved

- → Rejected

## Data Preprocessing

The dataset required several preprocessing steps:

Handling Missing Values

Non-standard missing value markers (?, NA, empty strings) were converted to NaN

Imputation strategy:

Numerical features → replaced with the column mean

Categorical features → replaced with the column mode

Encoding Categorical Variables

Categorical variables were converted using one-hot encoding

drop_first=True was used to avoid redundant features and multicollinearity

Feature Scaling

Numerical features were standardized using StandardScaler

Scaling was included inside a Pipeline to prevent data leakage

## Model Selection

A Logistic Regression classifier was chosen because:

It is well-suited for binary classification

It produces interpretable results

It is widely used in financial risk modeling

🔧 Model Training & Tuning

The dataset was split into training and test sets

A Pipeline was used to combine:

Feature scaling

Logistic Regression

GridSearchCV was applied to tune the C regularization parameter and solver

5-fold cross-validation was used to ensure robust performance

## Model Evaluation

The model was evaluated using:

Accuracy

Confusion Matrix

These metrics provide insight into how well the model distinguishes between approved and rejected applications.

## Results

The tuned Logistic Regression model achieved a high accuracy on the test set

The confusion matrix shows strong performance across both classes, indicating reliable classification

## Conclusion

This project demonstrates a complete machine learning pipeline for automating credit card approval decisions. It highlights best practices in preprocessing, model selection, and evaluation, and reflects techniques commonly used in real-world financial institutions.
