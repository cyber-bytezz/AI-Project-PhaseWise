<!-- Add your project logo/banner here -->
<p align="center">
  <img src="https://github.com/cyber-bytezz/IBM-AI_Phase3-4-5/assets/130319315/fe3f5931-2058-4e06-a7dc-fb710d3e65f0" alt="Project Logo" width="200">
</p>

# Data Analysis and Machine Learning Projects

> Welcome to the Developer Documentation for our Data Analysis and Machine Learning Projects! This repository contains a collection of projects showcasing data analysis and machine learning techniques.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Table of Contents

- [Project 1: Data Analysis - Tamil Nadu Companies](#project-1-data-analysis---tamil-nadu-companies)
  - [Overview](#overview)
  - [Code Snippets](#code-snippets)
- [Project 2: Machine Learning - Company Trend Prediction](#project-2-machine-learning---company-trend-prediction)
  - [Overview](#overview-1)
  - [Code Snippets](#code-snippets-1)
  - [Results](#results)
- [Developer's Guide](#developers-guide)
  - [Getting Started](#getting-started)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Project 1: Data Analysis - Tamil Nadu Companies

### Overview
This project involves the analysis of a dataset containing information about companies registered in Tamil Nadu. The analysis includes data cleaning, descriptive statistics, visualizations, grouping, and aggregation. Additionally, Principal Component Analysis (PCA) is performed to visualize the relationships between features.

#### Code Snippets:

- **Basic Information about the Dataset**
  ```python
  import pandas as pd
  df = pd.read_csv('Data_Gov_Tamil_Nadu.csv', encoding='ISO-8859-1')
  ```

- **Descriptive Statistics**
  ```python
  print("\nSummary Statistics for Numeric Columns:")
  print(df.describe())
  ```

- **Principal Component Analysis (PCA)**
  ```python
  from sklearn.decomposition import PCA
  features = df.select_dtypes(include=['float64', 'int64'])
  pca = PCA(n_components=2)
  principal_components = pca.fit_transform(features)
  # ... (rest of your code)
  ```

## Project 2: Machine Learning - Company Trend Prediction

### Overview
In this project, a machine learning model is built to predict the trend category of companies based on their financial attributes. The dataset is preprocessed, labeled, and split into training and testing sets. A RandomForestClassifier is used for the prediction, and hyperparameter tuning is performed to optimize the model's performance.

#### Code Snippets:

- **Data Preprocessing**
  ```python
  x = df.drop(['CORPORATE_IDENTIFICATION_NUMBER', 'COMPANY_NAME', 'COMPANY_STATUS', 'DATE_OF_REGISTRATION'], axis=1)
  y = df['REGISTRAR_OF_COMPANIES']
  ```

- **Label Encoding**
  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  df['REGISTRAR_OF_COMPANIES'] = le.fit_transform(df['REGISTRAR_OF_COMPANIES'])
  ```

- **Data Splitting and Model Training**
  ```python
  x = df[['AUTHORIZED_CAP', 'PAIDUP_CAPITAL', 'INDUSTRIAL_CLASS']]
  y = df['REGISTRAR_OF_COMPANIES']
  
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  
  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  ```

- **Model Evaluation**
  ```python
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  print(f"Accuracy: {accuracy}")
  print(f"Confusion Matrix:\n{conf_matrix}")
  ```

### Results

- **Project 1: Data Analysis**
  - Summary statistics provided insights into the dataset.
  - Visualizations enhanced understanding of the data.
  - PCA helped visualize the relationships between features.

- **Project 2: Company Trend Prediction**
  - Achieved an accuracy of 82.42% after hyperparameter tuning.

## Developer's Guide

### Getting Started

Provide steps for developers to get started with your project. Include information on prerequisites, installation instructions, and any setup needed.

### Usage

Explain how to use your project, including any important configuration options or environment variables.

### Contributing

Provide guidelines for other developers to contribute to your project. Include information on how to submit pull requests and report issues.

### License

This project is licensed under the [MIT License](LICENSE).
