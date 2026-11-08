# Machine Learning Classification Models - Wine Quality Dataset

## 📋 Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict wine quality based on physicochemical properties. The goal is to understand how different algorithms perform on the same dataset and create an interactive web application for model deployment and evaluation.

Wine quality prediction is important for the wine industry as it helps:
- Identify factors affecting wine quality
- Enable quality control in production
- Assist in pricing and marketing decisions
- Support decision-making in wine production optimization

## 📊 Dataset Description

**Dataset Name:** Wine Quality Dataset  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

### Dataset Overview
- **Total Samples:** 6,497 wine records (1,599 red + 4,898 white, combined into one dataset)
- **Total Features:** 12 (11 physicochemical properties + `wine_type`)
- **Target Variable:** `quality` (integer score 3-9)
- **Type:** Multi-class Classification Problem (7 classes)
- **Missing Values:** None

### Features (Input Variables)
1. **fixed acidity** - Amount of fixed acids in the wine
2. **volatile acidity** - Amount of volatile acids (acetic acid mainly)
3. **citric acid** - Amount of citric acid present
4. **residual sugar** - Sugar content after fermentation
5. **chlorides** - Salt content in the wine
6. **free sulfur dioxide** - Amount of free SO2
7. **total sulfur dioxide** - Total amount of SO2
8. **density** - Mass per unit volume of wine
9. **pH** - Acidity measure (0-14 scale)
10. **sulphates** - Potassium sulphate content (preservative)
11. **alcohol** - Alcohol content by volume
12. **wine_type** - `red` or `white` (encoded as red=0, white=1)

### Target Variable
- **quality:** Wine quality rating given by experts (3 to 9)

| Quality | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---------|---|---|---|---|---|---|---|
| Count   | 30 | 216 | 2138 | 2836 | 1079 | 193 | 5 |

### Dataset Characteristics
- **Feature Type:** 11 continuous numeric + 1 binary categorical
- **Imbalanced:** Yes - classes 5 and 6 make up ~77% of samples; classes 3 and 9 are very rare
- **Preprocessing:** `wine_type` encoded, all features standardized with `StandardScaler` (fit on training data only)
- **Train-Test Split:** 80/20 stratified split (`random_state=42`) -> 5,197 train / 1,300 test rows
- **test_data.csv:** the 1,300-row held-out test split (12 features + `quality`). Upload this file in the Streamlit app.

## 🔗 GitHub Repository Link

**Repository URL:** https://github.com/kummishra/bits_ml_assignment2.git

### Repository Structure
```
bits_ml_assignment2/
├── app.py                        # Streamlit web application
├── train_models.py               # Model training script
├── download_data.py              # Dataset download script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── test_data.csv                 # Held-out test split (1,300 rows)
├── data/
│   └── wine_quality.csv          # Full combined dataset (6,497 rows)
├── model/                        # Directory containing trained models
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── scaler.joblib
└── model_results.csv             # Model performance results
```

### Repository Contents
- ✅ Complete source code for all 5 classification models
- ✅ Requirements.txt with all dependencies
- ✅ Clear README.md with documentation
- ✅ Test data (CSV) used in experiments
- ✅ Trained model files for deployment
- ✅ Streamlit app for interactive testing

## 🤖 Models Used

All 5 models are trained on the same 80% training split and evaluated on the same 1,300-row held-out test set (`test_data.csv`).

| Model | Key settings |
|-------|--------------|
| Logistic Regression | `max_iter=1000`, lbfgs solver |
| Decision Tree | `max_depth=15`, Gini criterion |
| KNN | `n_neighbors=5`, Euclidean distance |
| Naive Bayes | `GaussianNB` (default) |
| Support Vector Machine | RBF kernel, `probability=True` |

Metric notes: Precision, Recall and F1 are **weighted averages** across the 7 classes (so weighted Recall equals Accuracy). AUC is the **weighted one-vs-rest** ROC AUC computed from `predict_proba`.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.5408 | 0.7229 | 0.5415 | 0.5408 | 0.5132 | 0.2707 |
| Decision Tree | **0.5900** | 0.7089 | **0.5876** | **0.5900** | **0.5875** | **0.3910** |
| KNN | 0.5585 | 0.7384 | 0.5441 | 0.5585 | 0.5489 | 0.3259 |
| Naive Bayes | 0.3215 | 0.5966 | 0.4207 | 0.3215 | 0.3621 | 0.0996 |
| Support Vector Machine | 0.5723 | **0.7605** | 0.5719 | 0.5723 | 0.5389 | 0.3211 |

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Weakest of the non-NB models (Accuracy 0.54, MCC 0.27). Its F1 (0.51) is below its accuracy, showing it mostly predicts the majority classes 5 and 6. A linear boundary cannot separate neighbouring quality levels well, though its AUC (0.72) shows the probability ranking is reasonable. |
| Decision Tree | Best Accuracy (0.59), F1 (0.59) and MCC (0.39). Non-linear splits capture feature interactions (e.g. alcohol x volatile acidity) and it predicts minority classes like 7 and 8 more often. But it has the 2nd-lowest AUC (0.71): leaf probabilities are almost 0/1, so its probability ranking is poor, and `max_depth=15` risks overfitting. |
| KNN | Mid-range (Accuracy 0.56, MCC 0.33) with the 2nd-best AUC (0.74). Gains from feature scaling and local neighbourhoods, but with k=5 the dense majority classes dominate the vote, hurting recall on rare classes. |
| Naive Bayes | Clearly the worst (Accuracy 0.32, MCC 0.10, AUC 0.60). The feature-independence assumption is badly violated (density is strongly tied to alcohol and residual sugar; free and total SO2 are correlated). Several features are also skewed rather than Gaussian, so the model over-predicts minority classes and Precision (0.42) far exceeds Recall (0.32). |
| Support Vector Machine | 2nd-best Accuracy (0.57) and the **best AUC (0.76)**: the RBF kernel ranks class probabilities best. Its F1 (0.54) and MCC (0.32) are lower than the Decision Tree's because it favours classes 5 and 6 and rarely predicts the minority classes. |
| **Overall winner** | **Decision Tree.** It leads on Accuracy, Precision, Recall, F1 and MCC, and MCC is the most reliable single metric for this imbalanced multi-class problem. SVM is the best choice if well-ranked probabilities (AUC) matter more. No model passes ~59% accuracy, which suggests the physicochemical features only partly explain the experts' subjective quality scores. |

## 🚀 How to Run

### Prerequisites
- Python 3.13 (the pinned package versions in requirements.txt need Python 3.11+)
- pip package manager

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kummishra/bits_ml_assignment2.git
   cd bits_ml_assignment2
   ```

2. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset**
   ```bash
   python download_data.py
   ```
   This downloads the red and white Wine Quality files from UCI and saves the combined data as `data/wine_quality.csv`

5. **Train Models**
   ```bash
   python train_models.py
   ```
   This will:
   - Load `data/wine_quality.csv` and split it 80/20
   - Save the held-out 20% as `test_data.csv`
   - Train all 5 classification models
   - Calculate evaluation metrics
   - Save trained models to `model/` directory
   - Display performance comparison

6. **Run Streamlit Application**
   ```bash
   streamlit run app.py
   ```
   The application will open in your default browser at `http://localhost:8501`

## 🌐 Streamlit Web Application

### Features

The deployed Streamlit application includes:

#### 📊 Model Comparison Tab
- Compare performance metrics of all 5 models on the uploaded test data
- View detailed comparison table with Accuracy, AUC, Precision, Recall, F1, and MCC
- Visualize metrics with bar charts and line plots
- Identify best-performing models at a glance

#### 🎯 Individual Predictions Tab
- All 6 metrics for the model chosen in the sidebar dropdown
- Per-class classification report

#### 📈 Confusion Matrix Tab
- Confusion matrix heatmap for the model chosen in the sidebar dropdown

#### 📋 Dataset Information Tab
- Dataset statistics and overview
- Class distribution visualization
- Data preview and summary statistics
- Feature information

### Usage Instructions

1. **Upload Test Data:**
   - Use the sidebar to upload a test CSV (e.g. `test_data.csv` from this repo)
   - If nothing is uploaded, the bundled `test_data.csv` is used
   - CSV must contain all 12 feature columns and the `quality` target column; `wine_type` may be `red`/`white` or 0/1
   - The saved models are only loaded and used for prediction; nothing is retrained

2. **Select Model:**
   - Pick a model from the sidebar dropdown for detailed metrics and the confusion matrix

3. **Interact with Results:**
   - Expand model sections to see detailed metrics
   - View visualizations of model performance
   - Analyze confusion matrices

4. **Export Results:**
   - Take screenshots of metrics and visualizations
   - Use table data for reports

## 📱 Streamlit Community Cloud Deployment

### Live Application Link
**Live app:** https://2026aj05545-bitsmlassignment2.streamlit.app/

### Deployment Steps (For Reference)

1. Push code to GitHub with:
   - `app.py`
   - `requirements.txt`
   - `test_data.csv`
   - `model/` directory with saved models

2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)

3. Sign in with GitHub account

4. Click "New App"

5. Select the repository and main branch

6. Streamlit automatically detects `app.py` and deploys

7. Share the public URL with evaluators

### Important Notes
- ✅ All dependencies listed in requirements.txt
- ✅ Test data included for immediate usage
- ✅ Pre-trained models for fast loading
- ✅ No retraining required on deployment
- ✅ Free tier compatible (models are small ~1MB)

## 📁 Project Files Description

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application |
| `train_models.py` | Model training and evaluation script |
| `download_data.py` | Dataset download utility |
| `requirements.txt` | Python package dependencies |
| `test_data.csv` | Held-out 20% test split (1,300 rows) for the app |
| `data/wine_quality.csv` | Full combined Wine Quality dataset |
| `README.md` | Project documentation (this file) |
| `model/logistic_regression.joblib` | Trained Logistic Regression model |
| `model/decision_tree.joblib` | Trained Decision Tree model |
| `model/knn.joblib` | Trained KNN model |
| `model/naive_bayes.joblib` | Trained Naive Bayes model |
| `model/support_vector_machine.joblib` | Trained SVM model |
| `model/scaler.joblib` | Feature scaler for preprocessing |
| `model_results.csv` | Exported model performance metrics |

## 📈 Evaluation Metrics Explained

### Accuracy
- **Definition:** Proportion of correct predictions out of total predictions
- **Formula:** (TP + TN) / (TP + TN + FP + FN)
- **Range:** 0-1 (0-100%)
- **Best for:** Balanced datasets

### AUC (Area Under the Curve)
- **Definition:** Probability that model ranks a random positive example higher than negative
- **Range:** 0-1 (0-1)
- **Best for:** Imbalanced datasets, threshold-independent evaluation
- **Interpretation:** 0.5 = random, 1.0 = perfect

### Precision
- **Definition:** Proportion of positive predictions that are actually positive
- **Formula:** TP / (TP + FP)
- **Range:** 0-1
- **Best for:** When false positives are costly

### Recall (Sensitivity)
- **Definition:** Proportion of actual positives correctly identified
- **Formula:** TP / (TP + FN)
- **Range:** 0-1
- **Best for:** When false negatives are costly

### F1 Score
- **Definition:** Harmonic mean of Precision and Recall
- **Formula:** 2 × (Precision × Recall) / (Precision + Recall)
- **Range:** 0-1
- **Best for:** Imbalanced datasets, balancing precision and recall

### MCC (Matthews Correlation Coefficient)
- **Definition:** Correlation coefficient between predicted and actual values
- **Range:** -1 to +1 (-1 = inverse prediction, 0 = random, +1 = perfect)
- **Best for:** Multi-class classification, imbalanced data
- **Advantage:** Single score considering all confusion matrix elements

## 🔧 Technologies Used

- **Python:** 3.13
- **Streamlit:** 1.67.0 - Web application framework
- **Scikit-learn:** 1.9.1 - Machine learning library (pinned: saved models must be loaded with the same version)
- **Pandas:** 3.1.1 - Data manipulation
- **NumPy:** 2.5.4 - Numerical computing
- **Matplotlib:** 3.11.2 - Data visualization
- **Seaborn:** 0.13.2 - Statistical visualization
- **Joblib:** 1.6.0 - Model serialization

## 📝 Notes and Considerations

### Data Preprocessing
- **Scaling:** StandardScaler applied to normalize features (mean=0, std=1)
- **Train-Test Split:** 80-20 split with stratification to maintain class distribution
- **Missing Values:** None present in the dataset
- **Outliers:** Not removed (wine quality properties can vary naturally)

### Model Configuration
- **Random State:** Set to 42 for reproducibility
- **Hyperparameters:** Mostly library defaults (see table above); no hyperparameter search was performed
- **Validation:** Used held-out test set (20% of data) for unbiased evaluation

### Limitations and Future Work
1. **Feature Engineering:** Additional features could improve accuracy
2. **Hyperparameter Tuning:** Grid search or Bayesian optimization could optimize models
3. **Ensemble Methods:** Combining models might improve performance
4. **Class Imbalance:** SMOTE or class weights could address imbalance
5. **Feature Selection:** Remove less important features for efficiency

## ✅ Assignment Checklist

- ✅ Wine Quality dataset: 12 features, 6,497 samples (meets the 12-feature / 500-instance minimum)
- ✅ Implemented all 5 classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM)
- ✅ Calculated all 6 evaluation metrics for each model
- ✅ requirements.txt with pinned dependencies
- ✅ test_data.csv (held-out split) and saved models in `model/`
- ✅ Streamlit app: CSV upload, model dropdown, metrics, confusion matrix + classification report
- ✅ App loads saved models and does not retrain on uploaded data

## 👤 Author

**Name:** Vivek Kumar  
**BITS ID:** 2026aj05545  
**Course:** M.Tech (AIML) - Machine Learning (BITS WILP)  
**Assignment:** 2  
**Submission Date:** 08-Nov-2026

## 📄 License

This project is created for educational purposes as part of BITS Pilani coursework.
