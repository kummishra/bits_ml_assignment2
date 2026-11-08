"""
Streamlit Web Application for ML Classification Models
Dataset: Wine Quality
Displays predictions and evaluation metrics for 5 different classification models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 ML Classification Models - Wine Quality Prediction")
st.markdown("""
This application demonstrates 5 different machine learning classification models 
trained on the **Wine Quality dataset**. Upload test data or use the sample data to see 
how each model performs.
""")

# Sidebar configuration
st.sidebar.title("📋 Model Configuration")
st.sidebar.write("---")

# Model selection
model_names = [
    'Logistic Regression',
    'Decision Tree',
    'KNN',
    'Naive Bayes',
    'Support Vector Machine'
]

selected_model = st.sidebar.selectbox(
    "Select Model:",
    model_names,
    help="Used for the Individual Predictions and Confusion Matrix tabs"
)
selected_models = [selected_model]

st.sidebar.write("---")

# Load trained models and scaler
@st.cache_resource
def load_models():
    """Load all trained models and scaler"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.joblib',
        'Decision Tree': 'model/decision_tree.joblib',
        'KNN': 'model/knn.joblib',
        'Naive Bayes': 'model/naive_bayes.joblib',
        'Support Vector Machine': 'model/support_vector_machine.joblib'
    }
    
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            models[model_name] = joblib.load(file_path)
        else:
            st.warning(f"Model file not found: {file_path}")
    
    # Load scaler
    scaler = None
    if os.path.exists('model/scaler.joblib'):
        scaler = joblib.load('model/scaler.joblib')
    
    return models, scaler


# Load models
try:
    models, scaler = load_models()
    st.sidebar.success(f"✓ Loaded {len(models)} models successfully")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    st.error("Please ensure all model files are available in the 'model/' directory")
    st.stop()

st.sidebar.write("---")

# Data upload section
st.sidebar.subheader("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

# Load default test data
test_data_path = 'test_data.csv'
if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.sidebar.success("✓ Custom test data loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        test_data = None
else:
    if os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
        st.sidebar.info("Using default test data (test_data.csv)")
    else:
        st.sidebar.warning("No test data available")
        test_data = None

# Main content
if test_data is not None and len(models) > 0 and scaler is not None:
    if 'quality' not in test_data.columns:
        st.error("Uploaded CSV must contain the target column 'quality'.")
        st.stop()
    
    feature_cols = list(scaler.feature_names_in_)
    missing_cols = [c for c in feature_cols if c not in test_data.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()
    
    X_test = test_data[feature_cols].copy()
    y_test = test_data['quality']
    
    # Same encoding as train_models.py (red=0, white=1); accepts text or already-encoded values
    if X_test['wine_type'].dtype == object or pd.api.types.is_string_dtype(X_test['wine_type']):
        X_test['wine_type'] = X_test['wine_type'].str.lower().map({'red': 0, 'white': 1})
    
    if X_test.isnull().any().any():
        st.error("Uploaded CSV contains missing or invalid values (wine_type must be 'red' or 'white').")
        st.stop()
    
    X_test_scaled = scaler.transform(X_test)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Model Comparison", "🎯 Individual Predictions", "📈 Confusion Matrix", "📋 Dataset Info"]
    )
    
    # Tab 1: Model Comparison
    with tab1:
        st.subheader("Model Performance Comparison (all 5 models)")
        
        if model_names:
            # Create comparison metrics
            comparison_data = []
            
            for model_name in model_names:
                if model_name in models:
                    model = models[model_name]
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Get prediction probabilities
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_scaled)
                    else:
                        y_pred_proba = None
                    
                    # Calculate metrics
                    from sklearn.metrics import (accuracy_score, precision_score, 
                                                recall_score, f1_score, 
                                                matthews_corrcoef, roc_auc_score)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    mcc = matthews_corrcoef(y_test, y_pred)
                    
                    # AUC Score
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except:
                        try:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        except:
                            auc = 0.0
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'AUC': auc,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'MCC': mcc
                    })
            
            # Display comparison table
            comparison_df = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            st.dataframe(
                comparison_df.style.format({
                    'Accuracy': '{:.4f}',
                    'AUC': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1': '{:.4f}',
                    'MCC': '{:.4f}'
                }).highlight_max(axis=0, color='lightgreen',
                                 subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
                width="stretch"
            )
            
            # Visualize metrics
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].plot(
                    kind='bar', ax=ax, width=0.8
                )
                ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
                ax.set_xlabel('Model')
                ax.set_ylabel('Score')
                ax.legend(loc='lower right')
                ax.set_ylim([0, 1])
                ax.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                comparison_df.set_index('Model')[['Accuracy', 'AUC', 'MCC']].plot(
                    kind='line', ax=ax, marker='o', linewidth=2, markersize=8
                )
                ax.set_title('Model Performance Trends', fontsize=14, fontweight='bold')
                ax.set_xlabel('Model')
                ax.set_ylabel('Score')
                ax.legend(loc='lower right')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
        else:
            st.info("Please select at least one model from the sidebar")
    
    # Tab 2: Individual Predictions
    with tab2:
        st.subheader("Detailed Model Predictions")
        
        if selected_models:
            for model_name in selected_models:
                if model_name in models:
                    with st.expander(f"📌 {model_name}", expanded=True):
                        model = models[model_name]
                        
                        # Predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Prediction probabilities
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test_scaled)
                        else:
                            y_pred_proba = None
                        
                        # Calculate metrics
                        from sklearn.metrics import (accuracy_score, precision_score, 
                                                    recall_score, f1_score, 
                                                    matthews_corrcoef, roc_auc_score)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        mcc = matthews_corrcoef(y_test, y_pred)
                        
                        # AUC Score
                        try:
                            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                        except:
                            try:
                                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                            except:
                                auc = 0.0
                        
                        # Display metrics in columns
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        col1.metric("Accuracy", f"{accuracy:.4f}")
                        col2.metric("AUC", f"{auc:.4f}")
                        col3.metric("Precision", f"{precision:.4f}")
                        col4.metric("Recall", f"{recall:.4f}")
                        col5.metric("F1 Score", f"{f1:.4f}")
                        col6.metric("MCC", f"{mcc:.4f}")
                        
                        # Classification report
                        st.write("**Classification Report:**")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format('{:.4f}'), width="stretch")
    
    # Tab 3: Confusion Matrix
    with tab3:
        st.subheader("Confusion Matrices")
        
        if selected_models:
            # Create a grid of confusion matrices
            n_models = len(selected_models)
            cols_per_row = 2
            n_rows = (n_models + cols_per_row - 1) // cols_per_row
            
            for i, model_name in enumerate(selected_models):
                if model_name in models:
                    if i % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[i % cols_per_row]:
                        model = models[model_name]
                        y_pred = model.predict(X_test_scaled)
                        
                        # Calculate confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   cbar_kws={'label': 'Count'},
                                   xticklabels=np.unique(y_test),
                                   yticklabels=np.unique(y_test))
                        ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
                        ax.set_ylabel('True Label')
                        ax.set_xlabel('Predicted Label')
                        st.pyplot(fig)
    
    # Tab 4: Dataset Info
    with tab4:
        st.subheader("Dataset Information")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(test_data))
        col2.metric("Total Features", X_test.shape[1])
        col3.metric("Target Classes", len(np.unique(y_test)))
        col4.metric("Class Distribution", f"{len(test_data)} records")
        
        st.write("**Class Distribution:**")
        class_dist = y_test.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        class_dist.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Target Variable Distribution', fontweight='bold')
        ax.set_xlabel('Wine Quality Class')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        st.write("**Dataset Preview:**")
        st.dataframe(test_data.head(10), width="stretch")
        
        st.write("**Dataset Statistics:**")
        st.dataframe(test_data.describe(), width="stretch")

else:
    st.warning("⚠️ Please ensure test data is available and models are loaded")
    if not os.path.exists(test_data_path):
        st.error("test_data.csv not found in the application directory")
    if len(models) == 0:
        st.error("No models loaded. Please ensure model files exist in the 'model/' directory")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='font-size: 12px; color: #666;'>
        Machine Learning Classification Models | Wine Quality Dataset<br>
        Built with Streamlit | Deployed on Streamlit Cloud
    </p>
</div>
""", unsafe_allow_html=True)
