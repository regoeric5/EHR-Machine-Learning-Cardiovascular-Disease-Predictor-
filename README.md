# EHR-Machine-Learning-Cardiovascular-Disease-Predictor-
> **Note**: This project was completed as part of independent research at Villanova University (CSC 5993). The full implementation code is currently under publication and will be made available following the publication process.
> 
> **Advisor**: Dr. Ebelechukwu Nwafor  

## 🎯 Project Overview

Cardiovascular Disease (CVD) affects over 116 million Americans, with a death occurring every 33 seconds. Early detection is critical to reducing CVD mortality rates. This project implements a machine learning system that predicts cardiovascular disease using only routine clinical data from Electronic Health Records (EHR), making it accessible for widespread clinical adoption.

Traditional cardiovascular disease prediction often relies on:
- Advanced biomarkers
- Medical imaging
- Specialized equipment

Our approach uses only tabular electronic health data from routine clinical observations, making CVD prediction more accessible and practical for real-world deployment. Importantly, all models were trained on a standard laptop without requiring specialized computing resources, GPU acceleration, or cloud infrastructure.

## 📊 Dataset

**Source**: MIMIC-IV v3.1 (Medical Information Mart for Intensive Care)
- Real-world Electronic Health Records from Beth Israel Deaconess Medical Center (BIDMC)
- De-identified patient data from hospital admissions

### Data Requirements
Each observation included in the study required:
- Unique hospital admission ID (HADM_ID)
- At least one laboratory test result
- At least one valid heart rate reading
- At least one prescription record

- 

### Features (49 Total)
- **Laboratory values** - Routine blood work and clinical tests
- **Vital signs** - Heart rate, blood pressure, temperature, etc.
- **Demographic information** - Age, gender, ethnicity
- **Comorbidity flags** - Pre-existing conditions and risk factors

"Random Forest Feature Importance Analysis"
- 

 Classification Task

The model predicts the presence of four types of cardiovascular disease:
1. **Heart Attack (Myocardial Infarction)
2. Ischemic Stroke
3. Heart Failure
4. Coronary Artery Disease (CAD)

**Output Classification**:
- `1` - Patient has one or more of the CVD conditions
- `0` - Patient has none of the CVD conditions

## Methodology

### Data Preprocessing Pipeline

1. **Feature Selection**
   - Excluded features with >30% null values
   - Maintained data quality while preserving sample size

2. **Missing Value Imputation**
   - Applied median imputation for remaining null values
   - Preserved statistical properties of the data

3. **Encoding**
   - Label encoding for categorical features
   - Converted all features to numeric representations

4. **Normalization**
   - Standard Scaler applied to all features
   - Ensured fair contribution across different measurement scales

5. **Train-Test Split**
   - 80% training data
   - 20% testing data

### Machine Learning Models

Six different models were implemented and evaluated:

1. **Logistic Regression Classifier** - Baseline linear model
2. **Random Forest Classifier** - Ensemble decision tree approach
3. **Support Vector Machine (SVM)** - Non-linear classification with kernel methods
4. **Extreme Gradient Boosting (XGBoost)** - Advanced gradient boosting framework
5. **Voting Classifier** - Hard/soft voting ensemble
6. **Meta-Learner Stacking Ensemble** - Multi-layer ensemble architecture

### Hyperparameter Optimization

- Grid Search Cross-Validation
- Systematically explored hyperparameter space
- Optimization metrics: ROC AUC and F1-Score
- Applied to all models for fair comparison

## 📈 Results

The models demonstrated strong predictive performance on routine clinical data:

- **Best performing model achieved high accuracy** on the test dataset
- **ROC AUC scores** indicated excellent discrimination ability
- **F1-Scores** showed balanced precision-recall performance
- Ensemble methods showed competitive or superior performance compared to individual models

*Detailed performance metrics and comparative analysis will be published in the forthcoming research paper.*

## ⚠️ Limitations

1. **Data Quality Dependencies**
   - Model performance is affected by errors in electronic health records
   - Inherits biases present in the source EHR system

2. **Feature Availability**
   - Many potentially informative features were excluded due to high null value percentages
   - Limited by data completeness in the MIMIC-IV dataset

3. **Temporal Information**
   - Current implementation does not utilize time series data
   - Missing temporal patterns that could enhance predictions

## 🚀 Future Work

1. **Time Series Integration**
   - Incorporate temporal patterns from patient history
   - Implement recurrent neural networks (RNN/LSTM)

2. **External Validation**
   - Test on additional hospital datasets
   - Assess generalizability across different populations

## 🛠️ Technology Stack

- **Python 3.8+**
- **scikit-learn** - Machine learning models and preprocessing
- **XGBoost** - Gradient boosting implementation
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Grid Search CV** - Hyperparameter optimization

**Department Head**: Professor Daniel Joyce

## 📄 Publication Status

**Current Status**: Under Publication

The full implementation, detailed methodology, and comprehensive results are currently being prepared for academic publication. The code repository will be made public following the publication process.


## 🙏 Acknowledgments

I would like to express my sincere gratitude to:
- **Dr. Ebelechukwu Nwafor** - Research advisor, for the invaluable guidance and support throughout this project
---


---

**Last Updated**: October 2025  
  README Available
