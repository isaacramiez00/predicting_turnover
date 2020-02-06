import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np

def plot_histograms():

    continous_features = ['satisfaction_level_percentage','last_evaluation_percentage','average_montly_hours']
    
    for feat in continous_features:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(x=stayed_df[feat], stacked=True, label='stayed', alpha=0.8)
        ax.hist(x=left_df[feat], stacked=True, label='left', alpha=0.8)
        ax.set_title(f'{feat}')
        ax.set_xlabel(f'{feat}')
        ax.set_ylabel('Frequency Count')
        plt.legend(loc='best')
        fig.tight_layout(pad=2)
        plt.savefig(f'{feat}.png')

def create_cat_percentage_df():

    left_df = turnover[turnover['left']==1]
    stayed_df = turnover[turnover['left']==0]

    cat_feature = ['number_project', 'time_spend_company_years',\
                   'Work_accident', 'promotion_last_5years', 'department', 'salary']

    for cat in cat_feature:
        left_df = left_df[cat_feature]
        stayed_df = stayed_df[cat_feature]

        # initiating the dataframe
        current = turnover[cat].value_counts()
        current_col = list(current.index)
        current_code = turnover[cat].value_counts()
        current_code_col = list(current_code.index)

        current_dict = {f'{cat}': current_col, 'code': current_code_col} # data for new dataframe
        current_df = pd.DataFrame(data=current_dict)

        # adding percentage convergance
        current_df['left'] = left_df[cat].value_counts()
        current_df['stayed'] = stayed_df[cat].value_counts()
        current_df['total_count'] = turnover[cat].value_counts()
        current_df['left_percentage'] = current_df['left'] / current_df['total_count']
        current_df.sort_values(by='left_percentage', axis=0, inplace=True)

        plot_side_by_side_percentage_barcharts(current_df)




def plot_side_by_side_percentage_barcharts(df):
 
    column = df.columns[0]
    labels = df[column]
    data = df['left_percentage']
    N = len(df)
    fig, ax = plt.subplots(figsize=(8,5))
    width = 0.8
    tickLocations = np.arange(N)
    ax.bar(tickLocations, data, width, linewidth=3.0, align='center')
    ax.set_xticks(ticks=tickLocations)
    ax.set_xticklabels(labels)
    ax.set_xlim(min(tickLocations)-0.6,\
                max(tickLocations)+0.6)
    ax.set_xlabel(f'{column}')
    ax.set_ylabel('Employee Percent Turnvover')
    # ax.set_yticks(np.linspace(0,max(),6))
    ax.yaxis.grid(True)
    ax.set_title(f'Employer Turnover by {column}')
    fig.tight_layout(pad=1)
    plt.savefig(f'Employer_Turnover_by_{column}.png')


if __name__=='__main__':
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

    create_cat_percentage_df()

    ## correlation matrix
    fig, ax = plt.subplots(figsize=(9,6))
    ax = sns.heatmap(turnover.corr())
    plt.savefig('correlation_matrix.png')
    plt.tight_layout(pad=4)

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


    ## continous-histograms
    plot_histograms()
    # plt.show()

    # random forest model w recall score metric
    rfc = RandomForestClassifier()
    y = turnover.pop('left').values
    X = turnover.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = rfc.predict(X_test)

    ## data leakage problem
    features = list(turnover.columns)
    feat_dict = {k: v for k, v in zip(features, rfc.feature_importances_)}

    for idx, feat in enumerate(features):
        data_leakage_feature = features.pop(idx)
        X = turnover[features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        feat_dict = {k: v for k, v in zip(features, rfc.feature_importances_)}
        print(feat_dict)
        print(f'data-leak-feature: {data_leakage_feature}')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn)
        print(f'fpr: {round(fpr,2)')

        train_accuracy = rfc.score(X_train, y_train)
        print(f'train-accuracy: {round(train_accuracy,2)}')

        test_accuracy = rfc.score(X_test, y_test)
        print(f'test-accuracy: {round(test_accuracy,2)} \n\n')