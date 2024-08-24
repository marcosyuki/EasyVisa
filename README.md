# EasyVisa: Visa Approval Prediction

## Business Overview
EasyVisa processes job certification applications for employers seeking to bring foreign workers into the United States and grants certifications in those cases where employers can demonstrate that there are not sufficient US workers available to perform the work at wages that meet or exceed the wage paid for the occupation in the area of intended employment. 

## Objective
The task at hand to analyze the data provided to facilitate the process of visa approvals by building a machine learning model and to recommend a suitable profile for the applicants for whom the visa should be certified or denied based on the drivers that significantly influence the case status.

## Key Skills
'Bagging', 'Random Forest and Boosting'

## Python Libraries

- **Warnings** (`import warnings`)
  - Suppress warnings (`warnings.filterwarnings('ignore')`)
- **NumPy** (`import numpy as np`)
- **Pandas** (`import pandas as pd`)
- **Matplotlib** (`import matplotlib.pyplot as plt`)
  - Inline plotting in Jupyter Notebooks (`%matplotlib inline`)
- **Seaborn** (`import seaborn as sns`)
- **Scikit-learn (sklearn)**
  - Data splitting and imputation:
    - `train_test_split` (`from sklearn.model_selection import train_test_split`)
    - `SimpleImputer` (`from sklearn.impute import SimpleImputer`)
  - Model building:
    - Ensemble classifiers:
      - `BaggingClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier` (`from sklearn.ensemble import ...`)
      - `XGBClassifier` (`from xgboost import XGBClassifier`)
      - `StackingClassifier` (`from sklearn.ensemble import StackingClassifier`)
    - Decision Tree Classifier: `DecisionTreeClassifier` (`from sklearn.tree import DecisionTreeClassifier`)
  - Model tuning and evaluation:
    - Metrics: `confusion_matrix`, `classification_report`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score` (`from sklearn.metrics import ...`)
    - Hyperparameter tuning: `GridSearchCV` (`from sklearn.model_selection import GridSearchCV`)

## Project Flowchart

<img width="898" alt="image" src="https://github.com/user-attachments/assets/f59ac966-56ea-4146-a9ab-2e844df21c98">

## Charts

![image](https://github.com/user-attachments/assets/b50cbb94-b27f-4511-a731-30a367a32b4a)

![image](https://github.com/user-attachments/assets/637fa326-6679-4fdb-ba9d-29c10ec21562)

![image](https://github.com/user-attachments/assets/86cba988-6f2c-47c0-8126-d4a2082a2d6c)

![image](https://github.com/user-attachments/assets/e57abd9a-b4da-4261-a6c0-0e451cbf49fa)

![image](https://github.com/user-attachments/assets/cc4519e0-d27d-4cfa-8eab-9239dafa1d37)
