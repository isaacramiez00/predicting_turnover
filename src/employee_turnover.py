import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from sklearn.metrics import recall_score, precision_score, precision_recall_curve
plt.style.use('ggplot')


'''
Under Construction - Not Ready for Deployment
'''

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


class EmployeeTurnoverDatasets():
    '''
    Class breaks down Turnover dataset and may return
    desired dataframe
    '''

    def __init__(self, df, drop_duplicates=True):

        if drop_duplicates:
            self.df.drop_duplicates(keep='first', inplace=True)
        
        self.drop_duplicates = drop_duplicates
        self.df = df
        self.features = list(self.df.columns)
        self.left_df = self.df[turnover['left']==1]
        self.stayed_df = self.df[turnover['left']==0]
        self.categorical_features = ['number_project', 'time_spend_company_years',\
                    'Work_accident', 'promotion_last_5years', 'department', 'salary']
        self.normalized_feature = None
        self.salary_code = None
        self.department_code = None
        self.continous_features = ['satisfaction_level_percentage','last_evaluation_percentage','average_montly_hours']
        self.encoded_df = None

    def normalize_categoricals(self, column):
        '''
        May only take categorical features of the turnover features:
        features =  ['number_project', 'time_spend_company_years',\
                    'Work_accident', 'promotion_last_5years', 'department', 'salary']
        call self.categorical_features to get list of possible columns to pass
        '''

        current_df = self.df.groupby(column).sum()
        total_val_counts = self.df[column].value_counts()

        if (column  == 'department') or (column == 'salary'):
            current_df.sort_values('left', axis=0, inplace=True)
            current_df = current_df.merge(total_val_counts, left_index=True, right_index=True)
            current_df.rename(columns={current_df.columns[-1]: 'total_count'}, inplace=True)

            if column == 'salary':
                current_df = current_df.reindex(index= ['low', 'medium', 'high'])
            
        else:
            total_val_counts.sort_index(ascending=True, inplace=True)
            current_df['total_count'] = total_val_counts.values   

        current_df['left_percentage'] = current_df['left'] / current_df['total_count']
        current_df['stayed_percentage'] = 1 - current_df['left_percentage']

        if column == 'department':
            current_df.sort_values('left_percentage', axis=0, inplace=True)

        elif column == 'Work_accident':
            current_df.rename(index={0:'No Accident', 1:'Accident'}, inplace=True)

        elif column == 'promotion_last_5years':
            current_df.rename(index={0:'No Promotion', 1:'Promotion'}, inplace=True)

        self.normalized_feature = current_df

        return current_df
            
    def encode_categorical_features(self, column):
        '''
        Used for Correlation Matrix and Random Forest Classifier
        Modeling; One-hot-encode categorical (object) features
        (Department or Salary column);
        return orders [column_df, self.df];
        df becomes the updated df with the encoded columns
        for department or salary
        '''

        self.encoded_df = self.df

        current = self.encoded_df[column].value_counts()
        current_col = list(current.index)
        self.encoded_df[column] = self.encoded_df[column].astype('category').cat.reorder_categories(current_col).cat.codes
        current_code = self.encoded_df[column].value_counts()
        current_code_col = list(current_code.index)
        current_dict = {f'{column}': current_col, 'code': current_code_col}
        df_name = f'{column}_df'
        df_name = pd.DataFrame(data=current_dict)

        if column == 'salary':
            self.salary_code = df_name
            return self.salary_code, self.encoded_df
        else:
            self.department_code = df_name
            return self.department_code, self.encoded_df

class EmployeeTurnoverVizualizations(EmployeeTurnoverDatasets):
    '''
    This class plots all the data vizualizations and inherits
    the EmployeeTurnoverDatasets class
    '''
    
    def __init__(self, df):
        super().__init__(df)

    def plot_histograms(self, feat):
        '''
        feat - continous feature from turnover dataset
        Best to only plot continous Features;
        These are the columns it can plot:
        ['satisfaction_level_percentage','last_evaluation_percentage','average_montly_hours']
        '''

        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(x=self.stayed_df[feat], stacked=True, label='stayed', alpha=0.8)
        ax.hist(x=self.left_df[feat], stacked=True, label='left', alpha=0.8)
        ax.set_title(f'{feat}')
        ax.set_xlabel(f'{feat}')
        ax.set_ylabel('Frequency Count')
        plt.legend(loc='best')
        fig.tight_layout(pad=2)
        plt.savefig(f'{feat}_new_style.png')

    def plot_side_by_side_percentage_barcharts(self, column):
        '''
        Works with normalized_categoricals() from EmployeeTurnoverDataset
        Plots a bar chart comparison of employees who stayed and left
        As a reminder the columns that best work with this method are:
        features =  ['number_project', 'time_spend_company_years',\
                    'Work_accident', 'promotion_last_5years', 'department', 'salary']
        '''
    
        stayed = self.normalized_feature['stayed_percentage'].values
        left = self.normalized_feature['left_percentage'].values

        labels = list(self.normalized_feature.index)

        fig, ax = plt.subplots(figsize=(10,5))
        width = 0.4
        xlocs = np.arange(len(stayed))
        ax.bar(xlocs-width, stayed, width, label='Stayed')
        ax.bar(xlocs, left, width, label='Left')
        ax.set_xticks(ticks=range(len(stayed)))
        ax.set_xticklabels(labels)
        ax.set_xlabel(f'{column}')
        ax.set_ylabel('Employee Percent Turnvover')
        ax.yaxis.grid(True)
        ax.legend(loc='best')
        ax.set_title(f'Employer Turnover by {column}')
        fig.tight_layout(pad=2)
        plt.savefig(f'Employer_Turnover_by_{column}_side_barcharts.png')

    def plot_corr_matrix(self,df):
        '''
        plots correlation matrix to
        find correlation among columns
        in turnover dataset
        (uses self.encoded_df)
        '''

        fig, ax = plt.subplots(figsize=(12,8))
        ax = sns.heatmap(self.encoded_df.corr())
        plt.tight_layout(pad=4)
        plt.savefig('correlation_matrix_newest.png')

    def plot_ROC_curve(self):
        '''
        plot roc curve based on given model
        paramater - "drop_duplicates" gives user
        the option to exploit data leakage
        before or after to compare
        ---UNDER CONSTRUCTION---
        '''

        # _, _, self.df = encode_cat_features(df)

        # print(confusion_matrix(y_test, y_pred))

        # roc curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        auc_ = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8,5))

        # roc curve plot
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr, tpr, label=f'{data_leakage_feature} = {round(recall,2)}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        if drop_duplicates:
            ax.set_title('ROC Curve comparing Features After Dropping Duplicates')
        else:
            ax.set_title('ROC Curve comparing Features Before Dropping Duplicates')
        ax.legend(loc='best') 
        # plt.savefig(f'ROC_Curve_Wout_{data_leakage_feature}_feature.png')

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
        recall_before_drop, _ = run_rfc_model(drop_duplicates=False, scores_only=True)
        recall_after_drop, _ = run_rfc_model(drop_duplicates=True, scores_only=True)
        
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

class EmployeeTurnoverClassifier():

    def run_rfc_model(drop_duplicates=False, scores_only=False):
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

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        if scores_only:
            return recall, precision
        elif drop_duplicates:
            return f'recall-score After leakage: {round(recall_before_drop,2)}\nprecision-score After leakage:{round(precision,2)}'
        else:
            return f'recall-score before leakage: {round(recall_before_drop,2)}\nprecision-score before leakage:{round(precision,2)}'


    def run_all_models():
        '''
        runs all models that were tested
        and returns recall score.
        '''
        pass

    def get_feature_importances(self):
        '''
        goes in model class
        '''
        rfc = RandomForestClassifier()
        y = self.df.pop('left').values
        recall_feature_leakage = {}

        for idx in range(len(self.df.columns)):
            features = self.features
            data_leakage_feature = features.pop(idx)
            X = self.df[features]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)

            feat_dict = {k: v for k, v in zip(features, rfc.feature_importances_)}

            # possible attributes
            # print(feat_dict)
            # print(f'data-leak-feature: {data_leakage_feature}')

    def get_recall_scores(self):
        '''
        recall scores for features
        ---UNDER CONSTRUCTION---
        '''
        
        # has to loop through each feature; somehow utilize the method above to perform stuff; In need of a lunch break
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        print(f'recall-score: {round(recall,2)}\nprecision-score: {round(precision,2)}')

        recall_feature_leakage[data_leakage_feature] = round(recall,2)


if __name__=='__main__':

    turnover = load_n_clean_data('../data/turnover.csv')
    t = EmployeeTurnoverDatasets(turnover)

    # data visuals
    # create_cat_percentage_df()
    # plot_histograms()
    # plot_ROC_curve(turnover) - still needs modification
    # plot_corr_matrix(turnover)

