import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from sklearn.metrics import recall_score
plt.style.use('ggplot')

'''
To whoever looks at this, I apoligize in advance for the terrible code... forgive me..i'm on it
'''

def plot_histograms():

    left_df = turnover[turnover['left']==1]
    stayed_df = turnover[turnover['left']==0]
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
        plt.savefig(f'{feat}_new_style.png')


def create_cat_percentage_df():

    left_df = turnover[turnover['left']==1]
    stayed_df = turnover[turnover['left']==0]

    cat_feature = ['number_project', 'time_spend_company_years',\
                   'Work_accident', 'promotion_last_5years', 'department', 'salary']

    for cat in cat_feature:

        current_df = turnover.groupby(cat).sum()
        total_val_counts = turnover[cat].value_counts()

        if (cat  == 'department') or (cat == 'salary'):
            current_df.sort_values('left', axis=0, inplace=True)
            current_df = current_df.merge(total_val_counts, left_index=True, right_index=True)
            current_df.rename(columns={current_df.columns[-1]: 'total_count'}, inplace=True)

            if cat == 'salary':
                current_df = current_df.reindex(index= ['low', 'medium', 'high'])
            
        else:
            total_val_counts.sort_index(ascending=True, inplace=True)
            current_df['total_count'] = total_val_counts.values   

        current_df['left_percentage'] = current_df['left'] / current_df['total_count']
        current_df['stayed_percentage'] = 1 - current_df['left_percentage']

        if cat == 'department':
            current_df.sort_values('left_percentage', axis=0, inplace=True)

        if cat == 'Work_accident':
            current_df.rename(index={0:'No Accident', 1:'Accident'}, inplace=True)

        if cat == 'promotion_last_5years':
            current_df.rename(index={0:'No Promotion', 1:'Promotion'}, inplace=True)

        plot_side_by_side_percentage_barcharts(current_df, column=cat)

def plot_ROC_curve():
    # random forest model w recall score metric

    turnover.drop_duplicates(keep='first', inplace=True)
    rfc = RandomForestClassifier()
    y = turnover.pop('left').values
    X = turnover.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    ## data leakage problem
    recall_feature_leakage = {}

    for idx in range(len(turnover.columns)):
        # breakpoint()
        features = list(turnover.columns)
        data_leakage_feature = features.pop(idx)
        X = turnover[features]
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        feat_dict = {k: v for k, v in zip(features, rfc.feature_importances_)}
        print(feat_dict)
        print(f'data-leak-feature: {data_leakage_feature}')
        
        recall = recall_score(y_test, y_pred)
        print(f'recall-score: {round(recall,2)}')
        recall_feature_leakage[data_leakage_feature] = recall
    
    print(X.shape)

    return recall_feature_leakage

        # print(confusion_matrix(y_test, y_pred))
        
        ## roc curve
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        # auc_ = auc(fpr, tpr)

        # fig, ax = plt.subplots(figsize=(8,5))

        # # roc curve plot
        # ax.plot([0, 1], [0, 1], 'k--')
        # ax.plot(fpr, tpr, label=f'RFC Recall Score= {round(recall,2)}')
        # ax.set_xlabel('False positive rate')
        # ax.set_ylabel('True positive rate')
        # ax.set_title(f'ROC curve With Out {data_leakage_feature} Feature')
        # ax.legend(loc='best')    
        # plt.savefig(f'ROC_Curve_Wout_{data_leakage_feature}_feature.png')



def plot_side_by_side_percentage_barcharts(df, column):
 
    stayed = df['stayed_percentage'].values
    left = df['left_percentage'].values

    labels = list(df.index)

    fig, ax = plt.subplots(figsize=(10,5))
    width = 0.4
    xlocs = np.arange(len(stayed))
    ax.bar(xlocs-width, stayed, width, label='Stayed')
    ax.bar(xlocs, left, width, label='left')
    ax.set_xticks(ticks=range(len(stayed)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(f'{column}')
    ax.set_ylabel('Employee Percent Turnvover')
    ax.yaxis.grid(True)
    ax.legend(loc='best')
    ax.set_title(f'Employer Turnover by {column}')
    fig.tight_layout(pad=2)
    plt.savefig(f'Employer_Turnover_by_{column}_side_barcharts.png')


if __name__=='__main__':
    # cleaning data 
    # breakpoint()

    turnover = pd.read_csv("../data/turnover.csv")
    salary = turnover['salary'].value_counts()
    salary_col = list(salary.index)
    # turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(salary_col).cat.codes I DID THIS FOR THE CORRMATRIX
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
    # turnover["department"] = turnover["department"].astype('category').cat.reorder_categories(department_col).cat.codes I DID THIS FOR CORRMATRIX
    # department = pd.get_dummies(turnover["department"])
    # turnover = turnover.drop(["department"], axis=1)
    department_code = turnover.department.value_counts()
    department_code_col = list(department_code.index)

    department_dict = {'department': department_col, 'code': department_code_col}
    department_df = pd.DataFrame(data=department_dict) # use for eda
    # breakpoint()
    # Data Visualization
    # create_cat_percentage_df()
    # plt.show()
    '''
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
    '''
    plot_histograms()
    plt.show()
    # plot_ROC_curve()

    # plotting fixing recall score
    # fig, ax = plt.subplots(figsize=(8,5))
    # ax.bar()

    # plt.show()

    # recall score before dropping data leakage
    # turnover.drop_duplicates(keep='first', inplace=True)
    # rfc = RandomForestClassifier()
    # y = turnover.pop('left').values
    # X = turnover.values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # rfc.fit(X_train, y_train)
    # y_pred = rfc.predict(X_test)


    # # print(f'data-leak-feature: {data_leakage_feature}')
    
    # recall_before_drop = recall_score(y_test, y_pred)
    # print(f'recall-score before leakage: {round(recall_before_drop,2)}')






    
    #######################################THIS IS AFTER EXPOSING DUPLICATES################################################### 
    
    
    
    
    # turnover = pd.read_csv("../data/turnover.csv")
    # salary = turnover['salary'].value_counts()
    # salary_col = list(salary.index)
    # turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(salary_col).cat.codes
    # salary_code = turnover.salary.value_counts()
    # salary_code_col = list(salary_code.index)

    # salary_dict = {'salary': salary_col, 'code': salary_code_col}
    # salary_df = pd.DataFrame(data=salary_dict) # use for eda

    # rename_columns = {'satisfaction_level': 'satisfaction_level_percentage',\
    #                 'last_evaluation': 'last_evaluation_percentage',\
    #                 'time_spend_company': 'time_spend_company_years',\
    #                 'sales': 'department'}
    # turnover.rename(columns=rename_columns, inplace=True)

    # department = turnover.department.value_counts()
    # department_col = list(department.index)
    # turnover["department"] = turnover["department"].astype('category').cat.reorder_categories(department_col).cat.codes

    
  
    # # after dropping data leakage
    # turnover.drop_duplicates(keep='first', inplace=True)

    # # set cleaned dataset to csv file
    # turnover.to_csv('../data/cleanedturnover.csv')
    # # turnover percent ratio  
    # print(len(turnover))
    # print(len(turnover[turnover['left']==1]))
    # turnover_perc_ratio = turnover['left'].value_counts()/len(turnover)*100
    # print(f'turnover percent ratio: \n {turnover_perc_ratio}')
    

    # y = turnover.pop('left').values
    # X = turnover.values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # rfc.fit(X_train, y_train)
    # y_pred = rfc.predict(X_test)


    
    # recall_after_drop = recall_score(y_test, y_pred)
    # print(f'recall-score after leakage: {round(recall_after_drop,2)}')

    # con_mat = confusion_matrix(y_test, y_pred)
    # tn, fp, fn, tp = con_mat.ravel()
    # print(f'tn: {tn}\n fp: {fp}\n fn: {fn}\n tp: {tp}')
    # print(f'''Confusion matrix after leakage: \n {con_mat}''')

    # recall_scores_arr = np.array([recall_before_drop, recall_after_drop])

    # labels = ['Recall Before', 'Recall After']
    # data = recall_scores_arr
    # N = len(recall_scores_arr)
    # fig, ax = plt.subplots(figsize=(8,5))
    # width = 0.8
    # tickLocations = np.arange(N)
    # ax.bar(tickLocations, data, width, linewidth=3.0, align='center')
    # ax.set_xticks(ticks=tickLocations)
    # ax.set_xticklabels(labels)
    # ax.set_xlim(min(tickLocations)-0.6,\
    #             max(tickLocations)+0.6)
    # ax.set_xlabel('Recall Scores')
    # ax.set_ylabel('Percentage')
    # ax.set_yticks(np.linspace(0,1,6))
    # ax.yaxis.grid(True)
    # ax.set_title('Recall Scores Before and After Data Leakage Exposed')
    # fig.tight_layout(pad=1)
    # plt.savefig('Recall_b_a_data_leakage.png')
    # plt.show()

    ## feature importance barplots
    # features = list(turnover.columns)
    # feat_dict = {k: v for k, v in zip(features, rfc.feature_importances_)}
    # # print(feat_dict)
    # # breakpoint()
    # imp_feat_df = pd.DataFrame([feat_dict])
    # imp_feat_df.sort_values(by=0, axis=1, inplace=True)  
    
    # labels = list(imp_feat_df.columns)
    # data = imp_feat_df.values
    # data = data.flatten()
    # y = np.arange(data.shape[0])

    # width = 0.8
    # fig, ax = plt.subplots(figsize=(8,5))
    # ax.barh(y, data, width, align='center')
    # ax.set_yticks(y)
    # ax.set_yticklabels(labels)
    # ax.xaxis.grid(True)
    # ax.set_ylabel('Feature Importance')
    # ax.set_xlabel('Percentage')
    # ax.set_title('Percentage by Feature Importance')
    # fig.tight_layout(pad=1)
    # plt.savefig('perc_by_feat_imp.png')
    # plt.show()

    # recall_feature_leakage = plot_ROC_curve()
    # recall_feature_df = pd.DataFrame([recall_feature_leakage])
