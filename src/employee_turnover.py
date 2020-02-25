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

def plot_histograms(df):

    left_df = df[turnover['left']==1]
    stayed_df = df[turnover['left']==0]
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


def create_cat_percentage_df(df):

    left_df = df[turnover['left']==1]
    stayed_df = df[turnover['left']==0]

    cat_feature = ['number_project', 'time_spend_company_years',\
                   'Work_accident', 'promotion_last_5years', 'department', 'salary']

    for cat in cat_feature:

        current_df = df.groupby(cat).sum()
        total_val_counts = df[cat].value_counts()

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

def plot_ROC_curve(df, drop_duplicates=False):
    '''
    plot roc curve based on given model
    '''

    _, _, df = encode_cat_features(df)

    if drop_duplicates:
        df.drop_duplicates(keep='first', inplace=True)
    
    rfc = RandomForestClassifier()
    y = df.pop('left').values
    # X = df.values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    recall_feature_leakage = {}

    for idx in range(len(df.columns)):
        features = list(df.columns)
        data_leakage_feature = features.pop(idx)
        X = df[features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        feat_dict = {k: v for k, v in zip(features, rfc.feature_importances_)}
        print(feat_dict)
        print(f'data-leak-feature: {data_leakage_feature}')
        
        recall = recall_score(y_test, y_pred)
        print(f'recall-score: {round(recall,2)}')
        recall_feature_leakage[data_leakage_feature] = round(recall,2)
    
        # print(X.shape)

        print(confusion_matrix(y_test, y_pred))

    breakpoint()
        # roc curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        auc_ = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8,5))

        # roc curve plot
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr, tpr, label=f'{data_leakage_feature} = {round(recall,2)}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve comparing Features')
        ax.legend(loc='best')    
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


def load_n_clean_data(filepath):
    '''
    loads and renames (cleans) the columns
    for the turnover file
    '''

    df = pd.read_csv(filepath)
    rename_columns = {'satisfaction_level': 'satisfaction_level_percentage',\
                    'last_evaluation': 'last_evaluation_percentage',\
                    'time_spend_company': 'time_spend_company_years',\
                    'sales': 'department'}
    df.rename(columns=rename_columns, inplace=True)
    return df

def encode_cat_features(df):
    '''
    one-hot-encode categorical (object) features
    (department and salary column)
    return orders [salary_df, department_df, df]
    df becomes the updated df with the encoded columns
    for department and salary
    '''

    cat_feature = ['salary', 'department']
    dfs = []

    for cat in cat_feature:
        current = df[cat].value_counts()
        current_col = list(current.index)
        df[cat] = df[cat].astype('category').cat.reorder_categories(current_col).cat.codes
        current_code = df[cat].value_counts()
        current_code_col = list(current_code.index)
        current_dict = {f'{cat}': current_col, 'code': current_code_col}
        df_name = f'{cat}_df'
        df_name = pd.DataFrame(data=current_dict)
        dfs.append(df_name)

    return dfs[0], dfs[1], df

def plot_corr_matrix(df):
    '''
    plots correlation matrix to
    find correlation among columns
    in turnover dataset
    '''

    _, _, encode_df = encode_cat_features(df)
    fig, ax = plt.subplots(figsize=(12,8))
    ax = sns.heatmap(encode_df.corr())
    plt.tight_layout(pad=4)
    plt.savefig('correlation_matrix_newest.png')

def plot_data_visualizations(df):
    '''
    plots all turnover eda
    and model evaluations
    '''

    create_cat_percentage_df() # edit
    plot_histograms()
    plot_ROC_curve()
    plot_percentage_comparison(df)
    # STILL NEED TO UPDATE ROC CURVE, PROFIT CURVE, PARTIAL DEPEDENDENCE, CONFUSION MATRIX, TEST/TRAIN VAL SCORES
    pass 

def run_rfc_model(drop_duplicates=False):
    '''
    runs the rfc model and returns recall score.
    Paramater gives user the option to drop duplicates
    found in the data to compare scores.
    '''

    if drop_duplicates:
        df.drop_duplicates(keep='first', inplace=True)

    rfc = RandomForestClassifier()
    y = df.pop('left').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    recall_before_drop = recall_score(y_test, y_pred)
    return f'recall-score before leakage: {round(recall_before_drop,2)}'

def run_all_models():
    '''
    runs all models that were tested
    and returns recall score.
    '''
    pass

def plot_percentage_comparison(df):
    '''
    simple bar plot returning the turnover ratio
    '''
    df['left_percentage'] = df['left'].value_counts()/ len(df)
    df['stayed_percentage'] = 1 - df['left_percentage']

    # simple bar plot to compare ratio
    pass

def plot_confusion_matrix():
    '''
    plot confusion matrix
    '''

    con_mat = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = con_mat.ravel()
    print(f'tn: {tn}\n fp: {fp}\n fn: {fn}\n tp: {tp}')
    print(f'''Confusion matrix after leakage: \n {con_mat}''')
    # create plot
    pass

def plot_feat_importances():
    '''
    plots feature importance of rfc model
    '''

    features = list(turnover.columns)
    feat_dict = {k: v for k, v in zip(features, rfc.feature_importances_)}

    imp_feat_df = pd.DataFrame([feat_dict])
    imp_feat_df.sort_values(by=0, axis=1, inplace=True)  
    
    labels = list(imp_feat_df.columns)
    data = imp_feat_df.values
    data = data.flatten()
    y = np.arange(data.shape[0])

    width = 0.8
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(y, data, width, align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.xaxis.grid(True)
    ax.set_ylabel('Feature Importance')
    ax.set_xlabel('Percentage')
    ax.set_title('Percentage by Feature Importance')
    fig.tight_layout(pad=1)
    plt.savefig('perc_by_feat_imp.png')

def plot_recall_scores():
    '''
    plots recall scores in bar chart
    '''

    recall_scores_arr = np.array([recall_before_drop, recall_after_drop])

    labels = ['Recall Before', 'Recall After']
    data = recall_scores_arr
    N = len(recall_scores_arr)
    fig, ax = plt.subplots(figsize=(8,5))
    width = 0.8
    tickLocations = np.arange(N)
    ax.bar(tickLocations, data, width, linewidth=3.0, align='center')
    ax.set_xticks(ticks=tickLocations)
    ax.set_xticklabels(labels)
    ax.set_xlim(min(tickLocations)-0.6,\
                max(tickLocations)+0.6)
    ax.set_xlabel('Recall Scores')
    ax.set_ylabel('Percentage')
    ax.set_yticks(np.linspace(0,1,6))
    ax.yaxis.grid(True)
    ax.set_title('Recall Scores Before and After Data Leakage Exposed')
    fig.tight_layout(pad=1)
    plt.savefig('Recall_b_a_data_leakage.png')

if __name__=='__main__':

    turnover = load_n_clean_data('../data/turnover.csv')

    # data visuals
    # create_cat_percentage_df()
    # plot_histograms()
    plot_ROC_curve(turnover)
    # plot_corr_matrix(turnover)

