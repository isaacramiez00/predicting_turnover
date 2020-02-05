import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import numpy as np


# cleaning data 
turnover = pd.read_csv("../data/turnover.csv")
salary = turnover['salary'].value_counts()
salary_col = list(salary.index)
turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(salary_col).cat.codes
salary_code = turnover.salary.value_counts()
salary_code_col = list(salary_code.index)

salary_dict = {'salary': salary_col, 'code': salary_code_col}
salary_df = pd.DataFrame(data=salary_dict) # use for eda

rename_columns = {'satisfaction_level': 'satisfaction_level_percentage',\
                  'last_evaluation': 'last_evaluation_percentage',\
                  'time_spend_company': 'time_spend_company_years',\
                  'sales': 'department'}
turnover.rename(columns=rename_columns, inplace=True)

department = turnover.department.value_counts()
department_col = list(department.index)
turnover["department"] = turnover["department"].astype('category').cat.reorder_categories(department_col).cat.codes
# department = pd.get_dummies(turnover["department"])
# turnover = turnover.drop(["department"], axis=1)
department_code = turnover.department.value_counts()
department_code_col = list(department_code.index)

department_dict = {'department': department_col, 'code': department_code_col}
department_df = pd.DataFrame(data=department_dict) # use for eda


explained_encoding_values_df = turnover[['department', 'salary']]
# explained_encoding_values_df['total']