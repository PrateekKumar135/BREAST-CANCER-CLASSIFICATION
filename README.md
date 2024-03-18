# Breast Cancer Classification using Machine Learning

This project focuses on developing a robust machine learning model for classifying breast cancer cases based on the Wisconsin Breast Cancer Diagnostic (WBCD) dataset. The goal is to build an accurate and reliable system that can assist medical professionals in early detection and diagnosis of breast cancer.

## Table of Contents

- [Dataset](#dataset)
- [Approach](#approach)
- [Models](#models)
- [Results](#results)



## Dataset

The dataset used in this project is the Wisconsin Breast Cancer Diagnostic (WBCD) dataset, which is a widely-used benchmark dataset for breast cancer classification tasks. It contains features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. The dataset consists of 569 instances, with 212 instances of malignant cases and 357 instances of benign cases.

## Approach

The project follows a systematic approach to building and evaluating machine learning models for breast cancer classification:

1. **Data Preprocessing**: Handling missing values, dropping unnecessary columns, and splitting the dataset into features and labels.
2. **Feature Engineering**: Scaling and dimensionality reduction techniques, such as Standard Scaler and Principal Component Analysis (PCA), are applied to the feature set.
3. **Model Selection**: Various machine learning models, including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Voting Ensemble, Gradient Boosting, AdaBoost, and Perceptron Neural Network, are explored and evaluated.
4. **Model Evaluation**: Repeated stratified cross-validation is employed to robustly assess the performance of the models, considering metrics such as accuracy, precision, and recall.

## Models

The following machine learning models have been implemented and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Voting Ensemble (Logistic Regression, KNN, SVM)
- Gradient Boosting
- AdaBoost
- Perceptron Neural Network

## Results

The results of the evaluated models are presented in the project report, including their respective accuracy, precision, and recall scores. The best-performing model is highlighted, along with its performance metrics and analysis.

