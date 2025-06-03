import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier, 
                             GradientBoostingClassifier, AdaBoostClassifier,
                             ExtraTreesClassifier, BaggingClassifier,
                             StackingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
import pickle

data = pd.read_csv('loan_prediction_dataset1.csv')

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
data['Dependents'] = data['Dependents'].replace({'3+': 3}).astype(float)

X = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
          'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
          'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
y = data['Loan_Status']

num_imputer = SimpleImputer(strategy='mean')
X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = \
    num_imputer.fit_transform(X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']] = \
    cat_imputer.fit_transform(X[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = [
    ('LDA', LinearDiscriminantAnalysis()),
    ('QDA', QuadraticDiscriminantAnalysis()),
    ('XGBoost', XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=200, max_depth=5)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42, C=0.1)),
    ('SVM', SVC(probability=True, random_state=42, C=1.0, kernel='rbf')),
    ('KNN', KNeighborsClassifier(n_neighbors=7)),
    ('Naive Bayes', GaussianNB()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42, max_depth=5)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.1)),
    ('AdaBoost', AdaBoostClassifier(random_state=42, n_estimators=100)),
    ('Extra Trees', ExtraTreesClassifier(random_state=42, n_estimators=100)),
    ('Bagging', BaggingClassifier(random_state=42, n_estimators=50)),
    ('CatBoost', CatBoostClassifier(random_state=42, verbose=0, iterations=200)),
    ('HistGradientBoosting', HistGradientBoostingClassifier(random_state=42, max_iter=200)),
    ('MLP', MLPClassifier(random_state=42, hidden_layer_sizes=(100,50), early_stopping=True))
]

print("\n=== Individual Model Performance on Test Set ===")
model_results = []
for name, model in models:
    if name in ['LDA', 'QDA', 'SVM', 'KNN', 'Logistic Regression', 'MLP']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    model_results.append((name, acc, model))
    print(f"{name:<25} Accuracy: {acc:.4f}")

model_results.sort(key=lambda x: x[1], reverse=True)

print("\n=== Models Ranked by Accuracy ===")
for name, acc, _ in model_results:
    print(f"{name:<25} Accuracy: {acc:.4f}")

print("\n=== Creating Ensemble Models ===")

first_level_models = model_results[:5]

final_estimator = LogisticRegression()

stacked_model = StackingClassifier(
    estimators=[(name, model) for name, _, model in first_level_models],
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

voting_model = VotingClassifier(
    estimators=[(name, model) for name, _, model in model_results],
    voting='soft',
    n_jobs=-1
)

stacked_model.fit(X_train, y_train)
y_pred_stacked = stacked_model.predict(X_test)
acc_stacked = accuracy_score(y_test, y_pred_stacked)
print(f"\nStacked Model Accuracy: {acc_stacked:.4f}")
print("Stacked Model Classification Report:")
print(classification_report(y_test, y_pred_stacked))

voting_model.fit(X_train_scaled, y_train)
y_pred_voting = voting_model.predict(X_test_scaled)
acc_voting = accuracy_score(y_test, y_pred_voting)
print(f"\nVoting Model Accuracy: {acc_voting:.4f}")
print("Voting Model Classification Report:")
print(classification_report(y_test, y_pred_voting))

if acc_stacked > acc_voting:
    final_model = stacked_model
    print("\n=== Final Model Selected: Stacked Model ===")
    print("Composition:")
    for name, _, _ in first_level_models:
        print(f"- {name}")
    print(f"Final estimator: {final_estimator.__class__.__name__}")
else:
    final_model = voting_model
    print("\n=== Final Model Selected: Voting Model ===")
    print("Composition (all models):")
    for name, _, _ in model_results:
        print(f"- {name}")

with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel training complete! Saved:")
print("- model.pkl (ensemble model)")
print("- scaler.pkl (standard scaler)")
