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


# Data Visualization

## department eda deepdive
left_df = turnover[turnover['left']==1]
department_df['left'] = left_df['department'].value_counts()

stayed_df = turnover[turnover['left']==0]
department_df['stayed'] = stayed_df['department'].value_counts()

department_df['total_count'] = turnover['department'].value_counts()
department_df['left_percentage'] = department_df['left'] / department_df['total_count']
department_df.sort_values(by='left_percentage', axis=0, inplace=True)

labels = department_df['department']
data = department_df['left_percentage']
N = 10
fig, ax = plt.subplots(figsize=(8,5))
width = 0.8
tickLocations = np.arange(N)
ax.bar(tickLocations, data, width, linewidth=3.0, align='center')
ax.set_xticks(ticks=tickLocations)
ax.set_xticklabels(labels)
ax.set_xlim(min(tickLocations)-0.6,\
            max(tickLocations)+0.6)
ax.set_xlabel('Department')
ax.set_ylabel('Employee Percent Turnvover')
ax.set_yticks(np.linspace(0,0.5,6))
ax.yaxis.grid(True)
ax.set_title('Employer Turnover by Department')
fig.tight_layout(pad=1)
plt.savefig('Employer_Turnover_by_Department.png')



## Salary eda deepdive

left_df = turnover[turnover['left']==1]
salary_df['left'] = left_df['salary'].value_counts()

stayed_df = turnover[turnover['left']==0]
salary_df['stayed'] = stayed_df['salary'].value_counts()

salary_df['total_count'] = turnover['salary'].value_counts()
salary_df['left_percentage'] = salary_df['left'] / salary_df['total_count']
salary_df.sort_values(by='left_percentage', axis=0, inplace=True)

labels = salary_df['salary']
data = salary_df['left_percentage']
N = 3
fig, ax = plt.subplots(figsize=(8,5))
width = 0.8
tickLocations = np.arange(N)
ax.bar(tickLocations, data, width, linewidth=3.0, align='center')
ax.set_xticks(ticks=tickLocations)
ax.set_xticklabels(labels)
ax.set_xlim(min(tickLocations)-0.6,\
            max(tickLocations)+0.6)
ax.set_xlabel('Salary Ranking')
ax.set_ylabel('Employee Percent Turnvover')
ax.set_yticks(np.linspace(0,0.5,6))
ax.yaxis.grid(True)
ax.set_title('Employer Turnover by Salary Rank')
fig.tight_layout(pad=1)
plt.savefig('Employer_Turnover_by_Salary_rank.png')


