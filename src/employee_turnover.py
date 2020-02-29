import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from sklearn.metrics import recall_score, precision_score, precision_recall_curve, f1_score
# from sklearn.metrics import plot_precision_recall_curve
plt.style.use('seaborn')


'''
Under Construction - Not Ready for Deployment
'''

def load_n_clean_data(filepath):
    '''
    loads and renames (cleans) the columns
    for the turnover file
    '''

    df = pd.read_csv(filepath)
    df.drop_duplicates(keep='first', inplace=True)
    rename_columns = {'satisfaction_level': 'satisfaction_level_percentage',\
                    'last_evaluation': 'last_evaluation_percentage',\
                    'time_spend_company': 'time_spend_company_years',\
                    'sales': 'department'}
    df.rename(columns=rename_columns, inplace=True)
    return df


class EmployeeTurnoverDatasets():
    '''
    Class breaks down Turnover dataset and return
    desired dataframe
    1. Highly Recommend - Use the transform_df() last 
    especially for data vizualization
    '''

    def __init__(self, df):

        self.df = df
        self.left_df = self.df[self.df['left']==1]
        self.stayed_df = self.df[self.df['left']==0]
        self.categorical_features = ['number_project', 'time_spend_company_years',\
                    'Work_accident', 'promotion_last_5years', 'department', 'salary']
        self.continous_features = ['satisfaction_level_percentage','last_evaluation_percentage','average_montly_hours']
        self.salary_code = None
        self.department_code = None
        self.featurize_df = None
        self._encode_featurized_columns()

    def transform_df(self, columns=['department', 'salary']):
        '''
        Create Featurized dataframe to pass in model
        '''

        for column in columns:
            current = self.df[column].value_counts()
            current_col = list(current.index)
            self.df[column] = self.df[column].astype('category').cat.reorder_categories(current_col).cat.codes

        return None

    def get_feature_turnover_ratio(self, column):
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

        current_df = current_df.iloc[:,-2:]

        return current_df
            
    def _encode_featurized_columns(self, columns=['department', 'salary']):
        '''
        Used for Correlation Matrix and Random Forest Classifier
        Modeling; One-hot-encode categorical (object) features
        (Department or Salary column);
        return orders [column_df, self.df];
        df becomes the updated df with the encoded columns
        for department or salary
        '''

        for column in columns:
            current_code = self.df[column].value_counts().reset_index()
            current_code.reset_index(inplace=True)
            current_code.drop(f'{column}', axis=1, inplace=True)
            current_code.rename(columns={'level_0': 'code', 'index': f'{column}'}, inplace=True)

            if column == 'salary':
                self.salary_code = current_code
            else:
                self.department_code = current_code
        
        return None


class EmployeeTurnoverClassifier(EmployeeTurnoverDatasets):
    '''
    Allows us to run models on Turnover dataset;
    calls run_model when instantiated to then give back
    results
    '''

    def __init__(self, df, model):

        super().__init__(df)
        self.model = model
        self.recall = None
        self.precision = None
        self.feature_importance = None
        self.feature_recall_comparison = None
        self.feature_precision_comparison = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.X = None
        self.y = None
        self.features = None
        self.f1_score = None

    def run_model(self, hold_feature=None):
        '''
        runs the model and gets recall score.
        Paramater gives user the option to drop duplicates
        found in the data to compare scores.
        '''

        if hold_feature != None:
            dropped_feature = self.df.pop(hold_feature).values

        self.y = self.df.pop('left').values
        self.features = list(self.df.columns)
        self.X = self.df.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X=self.X_train, y=self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        self.feature_importance = {k: v for k, v in zip(self.features, self.model.feature_importances_)}

        self.recall = recall_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred)
        self.f1_score = f1_score(self.y_test, self.y_pred)    

    def compare_recall_scores(self):
        '''
        recall scores for features
        ---UNDER CONSTRUCTION---
        '''
        
        recall_feature_leakage = {}
        precision_feature_leakge = {}
        f1_feature_leakage = {}
        # breakpoint()
        for idx in range(len(self.features)):
            local_feature = ['satisfaction_level_percentage', 'last_evaluation_percentage',\
                    'number_project', 'average_montly_hours',\
                    'time_spend_company_years', 'Work_accident', 'promotion_last_5years',\
                    'department', 'salary']
            features = local_feature
            data_leakage_feature = features.pop(idx)
            X = self.df[features]
            X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
        
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall_feature_leakage[data_leakage_feature] = round(recall,2)
            precision_feature_leakge[data_leakage_feature] = round(precision,2)
            f1_feature_leakage[data_leakage_feature] = round(f1,2)

        self.feature_recall_comparison = recall_feature_leakage
        self.feature_precision_comparison = precision_feature_leakge
        self.feature_f1score_comparison = f1_feature_leakage

class EmployeeTurnoverVizualizations(EmployeeTurnoverClassifier):
    '''
    This class plots all the data vizualizations and inherits
    the EmployeeTurnoverDatasets class
    '''
    
    def __init__(self, df, model):
        super().__init__(df, model)
        self.labels =  ['Satisfaction Level Percentage',
                        'last Performance Score',
                        'Number Of Projects',
                        'Average Monthly Working Hours',
                        'Time Spent in Company in Years',
                        'Accident at Work',
                        'Promoted In The Last 5 years',
                        'Department',
                        'Salary']


    def plot_histograms(self):
        '''
        feat - continous feature from turnover dataset  
        Best to only plot continous Features;
        These are the columns it can plot:
        ['satisfaction_level_percentage','last_evaluation_percentage','average_montly_hours']
        '''

        columns = ['Satisfaction Level Percentage', 'last Performance Score', 'Average Monthly Working Hours']

        for feat, col in zip(self.continous_features, columns):
            fig, ax = plt.subplots(figsize=(8,5))
            ax.hist(x=self.stayed_df[feat], stacked=True, label='stayed', alpha=0.8)
            ax.hist(x=self.left_df[feat], stacked=True, label='left', alpha=0.8)
            ax.set_title(f'{col} Comparison')
            ax.set_xlabel(f'{feat}')
            ax.set_ylabel('Frequency Count')
            plt.legend(loc='best')
            fig.tight_layout(pad=2)
            plt.savefig(f'{feat}_new_style.png')

    def plot_feature_turnover_barcharts(self):
        '''
        Works with normalized_categoricals() from EmployeeTurnoverDataset
        Plots a bar chart comparison of employees who stayed and left
        As a reminder the columns that best work with this method are:
        features =  ['number_project', 'time_spend_company_years',\
                    'Work_accident', 'promotion_last_5years', 'department', 'salary']
        '''

        cleaned_labels = ['Number Of Projects', 'Time Spent in Company in Years',\
                  'Accident at Work', 'Promotion In The Last 5 years',\
                  'Department','Salary']
    
        for column, cleaned in zip(self.categorical_features, cleaned_labels):
            
            current_df = self.get_feature_turnover_ratio(column)
            stayed = current_df.loc[:,'stayed_percentage'].values
            left = current_df.loc[:,'left_percentage'].values

            labels = list(current_df.index)
            fig, ax = plt.subplots(figsize=(10,5))
            width = 0.4
            xlocs = np.arange(len(stayed))
            ax.bar(xlocs-width, stayed, width, label='Stayed')
            ax.bar(xlocs, left, width, label='Left')
            ax.set_xticks(ticks=range(len(stayed)))
            ax.set_xticklabels(labels)
            ax.set_xlabel(f'{cleaned}')
            ax.set_ylabel('Employee Percent Turnvover')
            ax.yaxis.grid(True)
            ax.legend(loc='best')
            ax.set_title(f'Employer Turnover by {cleaned}')
            fig.tight_layout(pad=2)
            plt.savefig(f'Employer_Turnover_by_{column}_side_barcharts.png')

    def plot_corr_matrix(self):
        '''
        plots correlation matrix to
        find correlation among columns
        in turnover dataset
        (uses self.encoded_df)
        '''

        fig, ax = plt.subplots(figsize=(12,8))
        ax = sns.heatmap(self.df.corr())
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             rotation_mode="anchor")
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

        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred, pos_label=1)
        auc_ = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr, tpr, label=f'AUC = {round(auc_,2)}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        if self.df.shape[0] == 11991:
            ax.set_title('ROC Curve comparing Features After Dropping Duplicates')
        else:
            ax.set_title('ROC Curve comparing Features Before Dropping Duplicates')
        ax.legend(loc='best') 
        plt.savefig(f'ROC_Curve.png')

    def plot_percentage_comparison(self):
        '''
        simple bar plot returning the turnover ratio
        '''

        percent_left = self.left_df.shape[0] / self.df.shape[0]
        percent_stayed = 1 - percent_left

        percent_comparison = np.array([percent_stayed, percent_left])

        labels = ['Stayed', 'Turnover']
        data = percent_comparison
        N = len(percent_comparison)
        fig, ax = plt.subplots(figsize=(8,5))
        width = 0.8
        tickLocations = np.arange(N)
        ax.bar(tickLocations, data, width, linewidth=3.0, align='center')
        ax.set_xticks(ticks=tickLocations)
        ax.set_xticklabels(labels)
        ax.set_xlim(min(tickLocations)-0.6,\
                    max(tickLocations)+0.6)
        ax.set_xlabel('Employee Status')
        ax.set_ylabel('Percentage')
        ax.set_yticks(np.linspace(0,1,6))
        ax.yaxis.grid(True)
        ax.set_title('Employee Turnover Comparison Status')
        fig.tight_layout(pad=1)
        plt.savefig('percent_comparison.png')

    def plot_confusion_matrix(self):
        '''
        plot confusion matrix
        --UNDER CONSTRUCTION--
        '''

        con_mat = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = con_mat.ravel()
        
        # return print(f'tn: {tn}\n fp: {fp}\n fn: {fn}\n tp: {tp}')
        return np.array([[tn, fp],
                         [fn, tp]])
        # print(f'''Confusion matrix after leakage: \n {con_mat}''')
        # STILL NEED TO PLOT

    def plot_feat_importances(self):
        '''
        plots feature importance of rfc model
        '''

        # breakpoint()
        imp_feat_df = pd.DataFrame([self.feature_importance])
        imp_feat_df.sort_values(by=0, axis=1, inplace=True)  

        
        labels = ['Promoted In The Last 5 Years', 'Accident at Work', 'Salary', 'Department',\
                  'last Performance Score', 'Number Of Projects', 'Time Spent in Company in Years',\
                  'Average Monthly Working Hours', 'Satisfaction Level Percentage']
        data = imp_feat_df.values
        data = data.flatten()
        y = np.arange(data.shape[0])

        width = 0.8
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(y, data, width, color='green', alpha=0.5, align='center')
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.xaxis.grid(True)
        ax.set_ylabel('Feature Importance')
        ax.set_xlabel('Percentage')
        ax.set_title('Percentage by Feature Importance')
        fig.tight_layout(pad=1)
        plt.savefig('perc_by_feat_imp.png')

    def plot_f1_score_comparison(self):
        '''
        plots the comparision of recall score when we remove
        a feature to try and vizualize data leakage
        '''

       
        labels = self.labels
        data = list(self.feature_f1score_comparison.values())
        N = len(data)
        fig, ax = plt.subplots(figsize=(10,5))
        width = 0.8
        tickLocations = np.arange(N)
        ax.bar(tickLocations, data, width, color='green', alpha=0.5, linewidth=3.0, align='center')
        ax.axhline(y=min(data), linestyle='--', color='red')
        ax.set_xticks(ticks=tickLocations)
        ax.set_xticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             rotation_mode="anchor")
        ax.set_xlim(min(tickLocations)-0.6,\
                    max(tickLocations)+0.6)
        ax.set_xlabel('F1 Scores')
        ax.set_ylabel('Percentage')
        ax.set_yticks(np.linspace(0,1,6))
        ax.yaxis.grid(True)
        ax.set_title('F1 Scores Comparing Features')
        fig.tight_layout(pad=1)
        plt.savefig('feature_F1_comparison.png')

    # def plot_f1_score(self):

    #     disp = plot_precision_recall_curve(self.model, self.X_test, self.y_test)
    #     disp.ax_.set_title('2-class Precision-Recall curve: '
    #                'AP={0:0.2f}'.format(average_precision))

def plot_f1_scores(before, after):
    '''
    run different classes (one before drop and one after drop)
    plots recall scores in bar chart
    '''

    f1_scores_arr = np.array([before, after])
    labels = ['F1 Before', 'F1 After']
    data = f1_scores_arr
    N = len(f1_scores_arr)
    fig, ax = plt.subplots(figsize=(8,5))
    width = 0.8
    tickLocations = np.arange(N)
    ax.bar(tickLocations, data, width, color='green', alpha=0.5, linewidth=3.0, align='center')
    ax.axhline(y=min(data), linestyle='--', color='red')
    ax.set_xticks(ticks=tickLocations)
    ax.set_xticklabels(labels)
    ax.set_xlim(min(tickLocations)-0.6,\
                max(tickLocations)+0.6)
    ax.set_xlabel('F1 Scores')
    ax.set_ylabel('Percentage')
    ax.set_yticks(np.linspace(0,1,6))
    ax.yaxis.grid(True)
    ax.set_title('F1 Scores Before and After Data Leakage Exposed')
    fig.tight_layout(pad=1)
    plt.savefig('F1_b_a_data_leakage.png')

def run_all_models():
    '''
    runs all/different models that were tested
    and returns recall score.
    to be continued.
    --UNDER CONSTRUCTION--
    '''
    pass    

def main():
    '''
    runs Everything function so below we can do one call
    '''
    pass

if __name__=='__main__':
    data = load_n_clean_data('../data/turnover.csv')
    turnover = EmployeeTurnoverVizualizations(data, RandomForestClassifier())
    
    # modeling
    # turnover.transform_df()
    # turnover.run_model()
    # turnover.compare_recall_scores()
    # turnover.plot_f1_score_comparison()
    # before_leakage = turnover.f1_score
    # after_leakage = turnover.feature_f1score_comparison['time_spend_company_years']
    # plot_f1_scores(before_leakage, after_leakage)
    # turnover.plot_feat_importances()
    # turnover.plot_histograms()
    turnover.plot_feature_turnover_barcharts()
    plt.show()

    #expose data leakage