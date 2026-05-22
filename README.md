Project Overview
This project implements a dual-track data mining framework that bridges structural data balancing with high-efficiency pattern recognition
.
Customer Churn Classification: Using a Telco dataset to predict customer retention
.
Market Basket Analysis: Utilizing a groceries dataset to optimize retail cross-selling strategies
.

--------------------------------------------------------------------------------
🔬 Section 1: Customer Churn Classification Pipeline
The Class Imbalance Problem
The raw telecom dataset exhibited a significant class skew that risked biasing classifiers toward high accuracy while failing to detect actual churn
.
No Churn (Class 0): 5,174 instances
Churn (Class 1): 1,869 instances
To resolve this, we employed SMOTE (Synthetic Minority Over-sampling Technique) to generate balanced training instances
.
Raw Target Imbalance
Resampled Training Balance
<img src="assets/class_dist_raw.png" width="400"/>
<img src="assets/smote_comparison.png" width="400"/>
Model Performance Evolution
We evaluated Logistic Regression, K-Nearest Neighbors (KNN), and Decision Trees
. The application of SMOTE dramatically amplified Recall, which is critical for proactive retention strategies
.
Classifier Confusion Matrices (e.g., Logistic Regression)
The processed models successfully migrated false negatives into true positives, rescuing the model's ability to identify churning accounts
.
Raw Model Performance
Processed (Balanced) Performance
<img src="assets/cm_lr_raw.png" width="400"/>
<img src="assets/cm_lr_proc.png" width="400"/>

--------------------------------------------------------------------------------
🛒 Section 2: Market Basket Analysis Optimization
We benchmarked the Apriori algorithm against FP-Growth (Frequent Pattern Growth) using groceries.csv
.
Performance Findings
While both algorithms produced identical association rules (7 optimized associations), their execution footprints differed significantly
:
Apriori Execution: ~0.142 seconds
.
FP-Growth Execution: ~0.422 seconds
.
Algorithmic Comparison
Confidence Distribution
<img src="assets/comparison_chart.png" width="400"/>
<img src="assets/confidence_distribution.png" width="400"/>
Key Association Rules
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
📂 Repository Structure
To keep the project organized, files are structured as follows
:
/assets: Contains all 20+ performance visualizations and charts.
/data: Contains groceries.csv and the Telco dataset.
LAB FINAL DM Menahil_Suleman.ipynb: The core Jupyter Notebook.
README.md: Project documentation.

--------------------------------------------------------------------------------
👥 Authors
Menahil Suleman (FA24-BDS-029
