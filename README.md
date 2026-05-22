## Project Overview
This project implements a dual-track data mining framework that bridges structural data balancing with high-efficiency pattern recognition
.
Customer Churn Classification: Using a Telco dataset to predict customer retention
.
Market Basket Analysis: Utilizing a groceries dataset to optimize retail cross-selling strategies
.

<img width="2752" height="1536" alt="project" src="https://github.com/user-attachments/assets/6ba2d816-e835-439d-9066-34d762e65ee7" />

--------------------------------------------------------------------------------
## 🔬 Section 1: Customer Churn Classification Pipeline
The Class Imbalance Problem
The raw telecom dataset exhibited a significant class skew that risked biasing classifiers toward high accuracy while failing to detect actual churn
.
No Churn (Class 0): 5,174 instances
Churn (Class 1): 1,869 instances
To resolve this, we employed SMOTE (Synthetic Minority Over-sampling Technique) to generate balanced training instances
.
## Raw Target Imbalance
<img width="884" height="583" alt="class_dist_raw" src="https://github.com/user-attachments/assets/6ed23719-37b0-4021-af13-1803c479f6bb" />

## Resampled Training Balance
<img width="1634" height="617" alt="smote_comparison" src="https://github.com/user-attachments/assets/5daa4bdb-7a1c-49a0-b11f-c7b1bddace8c" />

Model Performance Evolution
We evaluated Logistic Regression, K-Nearest Neighbors (KNN), and Decision Trees
. The application of SMOTE dramatically amplified Recall, which is critical for proactive retention strategies
.
Classifier Confusion Matrices (e.g., Logistic Regression)
The processed models successfully migrated false negatives into true positives, rescuing the model's ability to identify churning accounts
.
## Raw Model Performance
<img width="738" height="581" alt="cm_lr_proc" src="https://github.com/user-attachments/assets/d685d044-0202-496d-815a-83fcf47b8a04" />

## Processed (Balanced) Performance
<img width="705" height="581" alt="cm_lr_raw" src="https://github.com/user-attachments/assets/fafd5b33-734e-43f3-8653-58ce721ebdbd" />

--------------------------------------------------------------------------------

## 🛒 Section 2: Market Basket Analysis Optimization
We benchmarked the Apriori algorithm against FP-Growth (Frequent Pattern Growth) using groceries.csv
.
Performance Findings
While both algorithms produced identical association rules (7 optimized associations), their execution footprints differed significantly
:
Apriori Execution: ~0.142 seconds
.
FP-Growth Execution: ~0.422 seconds
.
## Algorithmic Comparison
<img width="1483" height="593" alt="comparison_chart" src="https://github.com/user-attachments/assets/58763097-05be-4a23-a8b0-4e59b1bd4ae6" />


## Confidence Distribution
<img width="1783" height="591" alt="confidence_distribution" src="https://github.com/user-attachments/assets/3138277a-adca-4c6f-899a-46066de2c72d" />

## Key Association Rules
Our analysis discovered strong purchasing correlations ranked by statistical Lift
:
Root Vegetables → Other Vegetables: Lift = 2.29
.
Root Vegetables → Whole Milk: Lift = 1.78
.

--------------------------------------------------------------------------------
🛠️ Tech Stack & Libraries
Language: Python
Data Processing: Pandas, NumPy
Machine Learning: Scikit-Learn, Imbalanced-Learn (SMOTE)
Association Mining: MLxtend
Visualization: Matplotlib, Seaborn

--------------------------------------------------------------------------------
👥 Authors
Menahil Suleman (FA24-BDS-029)
