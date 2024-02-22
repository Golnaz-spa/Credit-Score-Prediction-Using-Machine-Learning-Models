# Credit Score Prediction Using Machine Learning Models

## Purpose
The purpose of this code is to predict credit scores based on historical loan data. The code implements various machine learning models to analyze the dataset and predict whether a loan will be fully paid or not. 

## Steps

1. **Importing Libraries**: Necessary libraries such as pandas, numpy, seaborn, matplotlib, and sklearn are imported.

2. **Loading Dataset**: The dataset named `loan_data_2007_2014.csv` is loaded using pandas.

3. **Data Preprocessing**:
   - The dataset's structure and information are displayed.
   - Distribution of the `loan_status` feature is plotted.
   - Null values in each feature are counted.
   - Features with over 80% missing values are identified and dropped.
   - The target variable `good_bad` is created based on the `loan_status`.
   - Features are separated into predictor variables (X) and target variable (y).
   - The dataset is split into training and testing sets.

4. **Data Cleaning**:
   - Several cleaning procedures are applied to features like `emp_length` and date columns.
   - Categorical features are converted to numerical using factorization.
   - Irrelevant or redundant columns are dropped from the dataset.

5. **Model Implementation and Evaluation**:
   - Random Forest Classifier (RFC) and Support Vector Machine (SVM) models are initialized and trained using the training data.
   - The performance of each model is evaluated using accuracy, confusion matrix, classification report, cross-validation, and time taken for execution.
   - Receiver Operating Characteristic (ROC) curve, F1 score, precision, and recall are calculated for each model.
   - Grid search is performed for hyperparameter tuning of the XGBoost model.
   - The best model from the grid search is evaluated, and its feature importance is plotted.
   - Confusion matrix for the test data is visualized.

## Aims
The aims of this project are as follows:
1. Implement machine learning models to predict credit scores.
2. Evaluate the performance of each model using various metrics such as accuracy, precision, recall, F1 score, and ROC AUC score.
3. Perform data preprocessing and cleaning to prepare the dataset for modeling.
4. Explore feature importance and conduct hyperparameter tuning to optimize model performance.
5. Visualize evaluation metrics and model results for better interpretation and understanding.
