# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:46:51 2024

@author: jessi
"""
#SLE Final Project Task 1
# Jessica Ahner, Yeonsoo Lim, & Yichen Lin

# You have to run this in the console for the dataset: pip install ucimlrepo

import pandas as pd
from ucimlrepo import fetch_ucirepo  # Note: Check if this function is correctly available from ucimlrepo package
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error

# Step 1: Fetch the dataset
try:
    predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)
    X = predict_students_dropout_and_academic_success.data.features
    y = predict_students_dropout_and_academic_success.data.targets
except Exception as e:
    print(f"Error fetching dataset: {e}")

# Drop irrelevant columns
columns_to_drop = [
    'Nacionality', 'Displaced', 'Curricular units 1st sem (credited)', 
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (without evaluations)', 
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'International', 
    'Curricular units 2nd sem (without evaluations)'
]
X = X.drop(columns=columns_to_drop)

# Define categorical and numerical columns
categorical_cols = [
    'Marital Status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Mother\'s qualification', 'Father\'s qualification',
    'Mother\'s occupation', 'Father\'s occupation', 'Educational special needs', 
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)'
]
numerical_cols = [
    'Previous qualification (grade)', 'Admission grade', 'Unemployment rate',
    'Inflation rate', 'GDP'
]

# Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.10, random_state=42, stratify=y_encoded
)

# Define models (without Random Forest for now)
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=2.0, penalty='l2'),
    'SVM': SVC(random_state=42, C=0.2, kernel='linear'),
    'Linear Regression (Lasso)': Lasso(alpha=0.1, random_state=42)
}

# Random Forest with OOB scoring
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=400,
    max_depth=None,
    oob_score=True
)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Get the OOB score
oob_error = 1 - rf_model.oob_score_
#print(f"\nRandom Forest OOB Error: {oob_error:.4f}")

# Add Random Forest to the model dictionary
models['Random Forest (OOB)'] = rf_model

# Evaluate models using cross-validation (except Random Forest)
best_models = {}
for model_name, model in models.items():
    if model_name == 'Random Forest (OOB)':
        model_score = 1 - oob_error  # Use OOB score directly
    else:
        # Fit the model before cross-validation
        model.fit(X_train, y_train)

        # Choose scoring metric
        if model_name == 'Linear Regression (Lasso)':
            scoring_method = 'neg_mean_squared_error'
        else:
            scoring_method = 'accuracy'

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=10, scoring=scoring_method, n_jobs=-1
        )
        model_score = -min(cv_scores) if model_name == 'Linear Regression (Lasso)' else max(cv_scores)

    print(f"\nModel: {model_name}")
    if model_name == 'Random Forest (OOB)':
        print(f"OOB Score: {model_score:.4f}")
    else: 
        print(f"Cross-Validation Score: {model_score:.4f}")

    best_models[model_name] = model_score

# Select the best model based on cross-validation or OOB score
best_model_name, best_model_score = max(best_models.items(), key=lambda x: x[1])


# Test the models on the test set
model_errors = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    if model_name == 'Linear Regression (Lasso)':
        test_error = mean_squared_error(y_test, y_pred)  # MSE for regression
    else:
        test_error = 1 - accuracy_score(y_test, y_pred)  # Error as 1 - accuracy
    model_errors[model_name] = test_error
    print(f"\nTest Error for {model_name}: {test_error:.4f}")

# Find the best model based on test errors
best_model_name_final = min(model_errors, key=model_errors.get)
best_model_error_final = model_errors[best_model_name_final]

print(f"\nBest model based on CV and OOB is {best_model_name} with a score of {best_model_score:.4f}.")
print(f"The best model after comparing test errors is {best_model_name_final} with a generalization error of {best_model_error_final:.4f}.")


