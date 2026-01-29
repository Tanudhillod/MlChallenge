🎓 Student Performance Prediction & Analysis (Machine Learning Project)

This project focuses on exploratory data analysis (EDA), feature engineering, class imbalance handling, and machine learning model evaluation to predict student academic outcomes such as Dropout, Enrolled, and Graduate using the Student Performance dataset.

The project includes complete preprocessing, visualization, multiple ML models, hyperparameter tuning, resampling techniques, and ensemble learning.

📌 Project Objectives

Analyze student demographic and academic data

Handle missing values, duplicates, and outliers

Perform detailed EDA and visualization

Address class imbalance issues

Apply feature engineering and transformation

Train multiple ML classification models

Compare performance using Accuracy and F1-score

Implement stacking ensemble model

📂 Dataset Information

Dataset Name: student.csv

Target Variable:

Target

Dropout

Enrolled

Graduate

Example Features:

Age

Gender

Nationality

Admission Grade

Previous Qualification Grade

Scholarship Holder

Tuition Fees Status

Debtor Status

Marital Status

Educational Special Needs

🛠 Technologies & Libraries Used
Programming Language

Python 🐍

Data Processing & Visualization

Pandas

NumPy

Matplotlib

Seaborn

Machine Learning & Preprocessing

Scikit-learn

Imbalanced-learn (SMOTE, ADASYN, SMOTEENN, SMOTETomek)

XGBoost

⚙ Project Workflow
1️⃣ Data Loading & Cleaning

Loaded dataset using Pandas

Renamed inconsistent column names

Removed duplicate records

Checked missing values

2️⃣ Exploratory Data Analysis (EDA)

Performed:

Target class distribution visualization

Correlation heatmap

Univariate analysis

Bivariate analysis

Multivariate analysis (pairplot)

Plots include:

Count plots

Histograms

Box plots

Violin plots

Heatmaps

3️⃣ Outlier Handling

Used IQR Method

Detected outliers in Age column

Replaced extreme values with median

4️⃣ Feature Engineering

Implemented:

Age binning (categorical age groups)

One-hot encoding

Polynomial feature creation

Log transformation

Weak correlation feature removal

5️⃣ Handling Class Imbalance

Used multiple techniques:

Class Weight Balancing

SMOTE

ADASYN

Random Oversampling

SMOTEENN

SMOTETomek

6️⃣ Data Preprocessing

Label Encoding

Feature Scaling using StandardScaler

Stratified Train-Test Split

🤖 Machine Learning Models Used

The following models were trained and tuned using GridSearchCV:

Model	Technique
Logistic Regression	Balanced Class Weight
Decision Tree	Balanced Class Weight
Random Forest	Balanced Class Weight
Support Vector Machine	Balanced Class Weight
K-Nearest Neighbors	SMOTE
Naive Bayes	SMOTE
AdaBoost	SMOTE
Bagging Classifier	SMOTE
XGBoost	Default Balancing
🧠 Ensemble Learning
✅ Stacking Classifier

Combined base learners:

Logistic Regression

Decision Tree

KNN

Random Forest

With Logistic Regression as final estimator.

📊 Model Evaluation Metrics

Each model was evaluated using:

Accuracy Score

Weighted F1-Score

Classification Report

Confusion Matrix Visualization

📈 Visualization Outputs

The project generates:

Correlation Heatmaps

Class Distribution Charts

Feature Distribution Plots

Confusion Matrices

Pairplots

Boxplots and Violinplots

🚀 How To Run This Project
Step 1: Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
Step 2: Upload Dataset

Place student.csv in the same directory as the script.

Step 3: Run Script
python your_script_name.py

or open the notebook in Google Colab / Jupyter Notebook.

📌 Key Highlights

✔ End-to-end ML pipeline
✔ Advanced EDA visualization
✔ Feature engineering techniques
✔ Imbalanced dataset handling
✔ Hyperparameter tuning
✔ Ensemble stacking model
✔ Production-ready ML workflow

📄 Future Improvements

Add deep learning models

Deploy using Flask / Streamlit

Add SHAP explainability

Optimize model performance

Create dashboard visualization

👩‍💻 Author

Tanu Kumari Dhillod
B.Tech CSE | Machine Learning Enthusiast
IGDTUW
