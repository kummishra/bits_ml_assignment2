"""
Machine Learning Classification Models Training Script
Dataset: Wine Quality
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


WINE_TYPE_MAP = {'red': 0, 'white': 1}


def load_and_prepare_data(filepath='data/wine_quality.csv'):
    """Load and prepare the Wine Quality dataset"""
    df = pd.read_csv(filepath)
    df = df.dropna()
    
    # Fixed mapping so the Streamlit app can apply the same encoding
    df['wine_type'] = df['wine_type'].map(WINE_TYPE_MAP)
    
    X = df.drop(columns=['quality'])
    y = df['quality']
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}\n")
    
    return X, y


def train_and_evaluate_models(X, y):
    """Train all 5 models and evaluate them"""
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
    
    # Held-out split only; this is the file uploaded to the Streamlit app
    test_df = X_test.copy()
    test_df['quality'] = y_test
    test_df.to_csv('test_data.csv', index=False)
    print(f"Saved held-out test split to 'test_data.csv' ({len(test_df)} rows)\n")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.joblib')
    
    # Dictionary to store models and results
    models = {}
    results = {}
    
    # 1. Logistic Regression
    print("=" * 60)
    print("Training Logistic Regression...")
    print("=" * 60)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)
    
    lr_results = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr)
    models['Logistic Regression'] = lr_model
    results['Logistic Regression'] = lr_results
    joblib.dump(lr_model, 'model/logistic_regression.joblib')
    print()
    
    # 2. Decision Tree Classifier
    print("=" * 60)
    print("Training Decision Tree Classifier...")
    print("=" * 60)
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=15)
    dt_model.fit(X_train_scaled, y_train)
    y_pred_dt = dt_model.predict(X_test_scaled)
    y_pred_proba_dt = dt_model.predict_proba(X_test_scaled)
    
    dt_results = evaluate_model(y_test, y_pred_dt, y_pred_proba_dt)
    models['Decision Tree'] = dt_model
    results['Decision Tree'] = dt_results
    joblib.dump(dt_model, 'model/decision_tree.joblib')
    print()
    
    # 3. K-Nearest Neighbors Classifier
    print("=" * 60)
    print("Training K-Nearest Neighbors Classifier...")
    print("=" * 60)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)
    
    knn_results = evaluate_model(y_test, y_pred_knn, y_pred_proba_knn)
    models['KNN'] = knn_model
    results['KNN'] = knn_results
    joblib.dump(knn_model, 'model/knn.joblib')
    print()
    
    # 4. Naive Bayes Classifier
    print("=" * 60)
    print("Training Gaussian Naive Bayes Classifier...")
    print("=" * 60)
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    y_pred_nb = nb_model.predict(X_test_scaled)
    y_pred_proba_nb = nb_model.predict_proba(X_test_scaled)
    
    nb_results = evaluate_model(y_test, y_pred_nb, y_pred_proba_nb)
    models['Naive Bayes'] = nb_model
    results['Naive Bayes'] = nb_results
    joblib.dump(nb_model, 'model/naive_bayes.joblib')
    print()
    
    # 5. Support Vector Machine
    print("=" * 60)
    print("Training Support Vector Machine...")
    print("=" * 60)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)
    
    svm_results = evaluate_model(y_test, y_pred_svm, y_pred_proba_svm)
    models['Support Vector Machine'] = svm_model
    results['Support Vector Machine'] = svm_results
    joblib.dump(svm_model, 'model/support_vector_machine.joblib')
    print()
    
    # Print summary
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    print("=" * 80 + "\n")
    
    # Save results to CSV
    summary_df.to_csv('model_results.csv')
    print("Results saved to 'model_results.csv'")
    
    return models, results, scaler


def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics for a model"""
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # AUC Score (for multi-class, use 'ovr' - One-vs-Rest)
    try:
        auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    # Precision (weighted for multi-class)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Recall (weighted for multi-class)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # F1 Score (weighted for multi-class)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Print detailed metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")
    
    return {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    }


if __name__ == "__main__":
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Train and evaluate models
    models, results, scaler = train_and_evaluate_models(X, y)
    
    print("\n✓ All models trained and saved successfully!")
    print("✓ Models saved in 'model/' directory")
    print("✓ Ready for Streamlit deployment!")

