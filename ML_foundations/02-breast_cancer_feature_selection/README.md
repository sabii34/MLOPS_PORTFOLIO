📌 Breast Cancer Feature Selection Analysis

Industry-Level ML Foundations Project

📖 Project Overview

This project performs an in-depth feature selection analysis on the widely-used Breast Cancer Wisconsin Diagnostic Dataset.
Feature selection is a crucial preprocessing step in Machine Learning that improves:

Model interpretability

Training efficiency

Performance of downstream models

Reduction of noisy or redundant features

This project explores several industry-standard feature selection techniques without performing final model training, aligning with ML foundations and preprocessing best practices.

🎯 Objectives

Load and explore the Breast Cancer dataset

Perform feature inspection & exploratory data analysis (EDA)

Apply multiple filter, wrapper, and embedded feature selection methods

Compare feature-ranking outputs

Identify consistently selected features

Generate insights for downstream predictive modeling

🗂 Project Structure
02-breast_cancer_feature_selection/
│
├── data/
│   └── breast_cancer.csv               # Dataset (optional local copy)
│
├── notebooks/
│   └── breast_cancer_feature_selection.ipynb # Main analysis notebook
│
├── venv/                               # Virtual environment (ignored in Git)
│
└── README.md                           # Project documentation


Note: The dataset can be loaded directly from sklearn.datasets, but a CSV may optionally be stored under /data.

🔧 Tech Stack
Component	Tools Used
Language	Python 3.x
Notebook	Jupyter Notebook
Libraries	NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
Versioning	Git + GitHub
OS	Windows
📦 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/MLOPS_PORTFOLIO.git
cd MLOPS_PORTFOLIO/ML_foundations/02-breast_cancer_feature_selection

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Required Dependencies
pip install -r requirements.txt


If you do not have a requirements file, install manually:

pip install numpy pandas scikit-learn matplotlib seaborn jupyter

4️⃣ Launch Jupyter Notebook
jupyter notebook


Open:

notebooks/breast_cancer_feature_selection.ipynb

🔍 Feature Selection Methods Implemented

The notebook includes multiple categories of feature selection techniques:

🔹 1. Filter Methods
Method	Purpose
Variance Threshold	Remove constant/near-constant features
Correlation Analysis	Detect redundancy & target correlation
Chi-Square Test	Identify top categorical/informational features
Mutual Information	Capture nonlinear dependencies
🔹 2. Wrapper Method
Method	Description
Recursive Feature Elimination (RFE)	Iteratively removes least important features based on model performance (Logistic Regression as base estimator)
🔹 3. Embedded Methods
Method	Description
L1-Regularized Logistic Regression	Shrinks coefficients to zero for feature pruning
Random Forest Feature Importance	Ranks features based on impurity reduction
📊 Analysis Output

The notebook generates:

Summary tables of selected features

Feature importance rankings

Visualizations such as:

Correlation heatmap

Mutual information plot

Random Forest feature importance bar chart

Additionally, a comparison is made to identify common high-value features across multiple methods.

🧠 Key Insights

Some features appear consistently across filter, wrapper, and embedded techniques

Correlated features help identify redundancy

L1 regularization and Random Forest are strong indicators of feature usefulness

Removing low-variance features is computationally efficient

These findings help guide more efficient model development in follow-up machine learning tasks.

📌 Limitations

No supervised learning model is trained (per project requirements)

Results are specific to the Breast Cancer Wisconsin dataset

Correlation analysis assumes linear relationships

🚀 Future Enhancements

The project can be extended by adding:

Model training & evaluation pipeline

Comparison of performance before vs. after feature selection

Hyperparameter tuning

MLOps components (DVC, GitHub Actions CI/CD, MLflow tracking)

Docker containerization

Deployment-ready preprocessing pipeline

🧑‍💻 Author

saba shahbaz
Machine Learning / MLOps Practitioner
📧 sabashahbaz731@gmail.com

🔗 GitHub:https://github.com/sabii34/MLOPS_PORTFOLIO