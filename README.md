# Breast Cancer Classification & Model Validation Pipelines

This repository contains a complete end-to-end Machine Learning pipeline for binary classification of breast cancer data. The workflow implements data preprocessing, feature engineering, logistics regression modeling, and robust evaluation using advanced data partitioning methodologies.

---

## Pipeline Architecture & Workflow

The pipeline is systematically broken down into reproducible processing steps:

1. **Exploratory Data Analysis (EDA)**: Profiling the structural dimensions (`.shape`) and inspecting features (`.head()`).
2. **Data Transformation**:
   * **Label Encoding**: Converts target variable designations (`Class`) into machine-readable numeric formats.
   * **Standardization**: Applies Z-score normalization (`StandardScaler`) to variance-stabilize the independent features ($X$).
3. **Data Partitioning**: Splitting data into stratified training and validation sets.
4. **Model Training**: Fitting a traditional regularized `LogisticRegression` classifier.
5. **Robust Validation Protocols**: Evaluating model variance and generalization using multiple split techniques.

---

## Cross-Validation Frameworks

To evaluate baseline stability and guard against data-leakage or lucky partitions, this script implements two distinct validation approaches:

### 1. Repeated Validation Set Approach (Monte Carlo)
* Shuffles and splits the dataset iteratively **100 times** using unique random states (`random_state=i`).
* Calculates the moving average across all evaluations to establish a generalized accuracy baseline.

### 2. K-Fold Cross Validation
* Strategically partitions the feature matrix into $K=5$ contiguous folds.
* Systematically trains the model on $K-1$ folds while validating performance on the remaining holdout subset.

---

## Tech Stack & Dependencies

Ensure you have the following framework dependencies installed within your active python environment:

```bash
pip install numpy pandas scikit-learn
