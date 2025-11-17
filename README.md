# Employee Attrition Prediction
## Project Overview
This project focuses on building and evaluating machine learning models to predict employee attrition. Employee attrition, or turnover, is a significant concern for organizations, impacting productivity, morale, and costs. By identifying employees at risk of leaving, companies can implement targeted retention strategies.

## Goal
The primary goal of this project is to develop a predictive model that can accurately identify employees likely to attrit, with a strong emphasis on detecting the minority class (employees who leave), even if it means a trade-off in overall accuracy. This approach prioritizes early warning signals for HR interventions.

## Dataset
The dataset used for this analysis is sourced from Kaggle: IBM HR Analytics Employee Attrition & Performance. It contains 1470 entries with 35 features, including demographic information, job-related metrics, and the target variable Attrition (Yes/No).

## Methodology
The project followed a standard machine learning pipeline:

1.Data Loading & Initial Exploration: Loaded the dataset, performed initial checks for missing values, duplicates, and understood column data types. Irrelevant columns (EmployeeCount, EmployeeNumber, Over18, StandardHours) were dropped.

2.Exploratory Data Analysis (EDA):

Generated a correlation heatmap to visualize relationships between numerical features.
Analyzed the distribution of the target variable Attrition using a countplot, revealing a significant class imbalance.
Examined the distribution of Age by Attrition using a boxplot.
3.Data Splitting: The dataset was split into training and testing sets (80/20 ratio) using train_test_split with stratify=y to ensure that the proportion of Attrition (Yes/No) was maintained in both subsets.

4.Feature Preprocessing: A ColumnTransformer was used to apply different preprocessing steps to different feature types:

  Numerical Features: Scaled using StandardScaler.
  Categorical Features: One-hot encoded using OneHotEncoder.
5.Handling Class Imbalance: RandomOverSampler was applied to the training data only to balance the classes, ensuring the models had sufficient examples of the minority class ('Yes' attrition) to learn from.

6.Model Training & Evaluation: Three different classification models were trained and evaluated:

   K-Nearest Neighbors (KNN)
   Random Forest Classifier
   Logistic Regression
Each model was evaluated using accuracy, confusion matrix, and a classification report (including precision, recall, and f1-score) to understand its performance on both the majority and minority classes. Models were evaluated both with and without oversampling on the training data to observe the impact of imbalance handling.

## Key Findings
Class Imbalance: The dataset exhibited a significant class imbalance, with a much smaller number of employees who attrited ('Yes') compared to those who did not ('No').
Impact of Oversampling:
Models trained without oversampling showed high overall accuracy but very poor recall for the 'Yes' attrition class, indicating they were ineffective at identifying actual leavers.
Models trained with RandomOverSampler on the training data generally saw a decrease in overall accuracy. However, they exhibited a substantial improvement in recall for the 'Yes' attrition class, demonstrating a better ability to detect employees at risk of leaving. This trade-off was deemed acceptable and necessary given the project's goal.
Model Performance (with Oversampling):
K-Nearest Neighbors: Showed strong recall for the minority class.
Random Forest: Also improved recall, though less dramatically than KNN or Logistic Regression.
Logistic Regression: Achieved comparable recall to KNN for the minority class.
## Conclusion
This project successfully demonstrated that effectively handling class imbalance is crucial for building meaningful predictive models in domains like HR attrition. While overall accuracy can be misleading, focusing on metrics like recall for the minority class provides a more practical and actionable understanding of a model's utility in real-world scenarios, where identifying at-risk individuals is paramount.
