import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import numpy as np

def rmse(theta, thetahat):
	''' Compute Root-mean-squared-error '''
	return np.sqrt(np.mean((theta - thetahat) ** 2))

def plot_errors():
	''' Plot errors from test and training sets '''
	m = X.shape[1]
	err_test, err_train = [], []
	logistic = LogisticRegression()
	for ind in range(m):
		logistic.fit(X_train[:,:(ind+1)], y_train)

		train_pred = logistic.predict(X_train[:,:(ind + 1)])
		test_pred = logistic.predict(X_test[:,:(ind + 1)])

		err_test.append(rmse(test_pred, y_test))
		err_train.append(rmse(train_pred, y_train))

	x = range(1, m+1)
	plt.figure()
	plt.plot(x, err_test, label='Test error')
	plt.plot(x, err_train, label='Training error')
	plt.title('Errors')
	plt.ylabel('RMSE')
	plt.xlabel('Features')
	plt.legend()
	plt.show()

def stage_score_plot(estimator, X_train, y_train, X_test, y_test):
    '''
        Parameters: estimator: GradientBoostingRegressor or AdaBoostRegressor
                    X_train: 2d numpy array
                    y_train: 1d numpy array
                    X_test: 2d numpy array
                    y_test: 1d numpy array

        Returns: A plot of the number of iterations vs the MSE for the model for
        both the training set and test set.
    '''
    estimator.fit(X_train, y_train)
    name = estimator.__class__.__name__.replace('Regressor', '')
    learn_rate = estimator.learning_rate
    # initialize 
    train_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    test_scores = np.zeros((estimator.n_estimators,), dtype=np.float64)
    # Get train score from each boost
    for i, y_train_pred in enumerate(estimator.staged_predict(X_train)):
        train_scores[i] = mean_squared_error(y_train, y_train_pred)
    # Get test score from each boost
    for i, y_test_pred in enumerate(estimator.staged_predict(X_test)):
        test_scores[i] = mean_squared_error(y_test, y_test_pred)
    plt.plot(train_scores, alpha=.5, label="{0} Train - learning rate {1}".format(
                                                                name, learn_rate))
    plt.plot(test_scores, alpha=.5, label="{0} Test  - learning rate {1}".format(
                                                      name, learn_rate), ls='--')
    plt.title(name, fontsize=16, fontweight='bold')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)

if __name__=='__main__':

    turnover = pd.read_csv("../data/turnover.csv")
    turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(['low', 'medium', 'high']).cat.codes
    turnover.rename(columns={'sales': 'department'}, inplace=True)
    department = pd.get_dummies(turnover["department"])
    turnover = turnover.drop(["department"], axis=1)

    # # add dummy variables -- not sure if I need it
    # turnover = turnover.join(department)

    # plotting the correlation matrix
    # as seaborn is based on matplotlib, we need to use plt.show() to see the plot
    # fig, ax = plt.subplots(figsize=(8,5))
    # ax = sns.heatmap(turnover.corr())
    # plt.show()
    # plt.savefig('correlation_matrix.png')
    # plt.tight_layout()
    # after looking into the correlation matrix, I decided to reduce to 4 features to predict employee turnover



    # # the percentage of leavers
    # turnover['left'].value_counts()/len(turnover)

    # # Decision Tree Classifier
    features = ['satisfaction_level', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']
    top_feat = features[:2]
    dt = DecisionTreeClassifier(random_state=42)
    y = turnover.pop('left').values
    X = turnover[top_feat].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt.fit(X_train, y_train)
    training_accuracy = dt.score(X_train, y_train)
    train_error = 1.0 - training_accuracy
    print(f'training-accuracy: {round(training_accuracy,2)}')
    print(f'train-error: {round(train_error,2)}')

    y_pred = dt.predict(X_test)
    test_accuracy = dt.score(X_test, y_test)
    test_error = 1.0 - test_accuracy
    print(f'test-accuracy: {round(test_accuracy, 2)}')
    print(f'test-error: {round(test_error,2)}')

    # decision tree plot
    # plt.figure(1)
    # tree.plot_tree(dt.fit(X, y), filled=True)
    # plt.show()

    # feature_importances_
    feat_dict = {k: v for k, v in zip(features, dt.feature_importances_)} # satisfaction_level and time spend_in_company -> best feat

    # confusion matrix
    print(confusion_matrix(y_test, y_pred))
    
    # # roc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    fig, ax = plt.subplots(2,1, figsize=(8,5))

    # roc curve plot
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].plot(fpr, tpr, label=f'DT; AUC = {round(auc_,2)}')
    ax[0].set_xlabel('False positive rate')
    ax[0].set_ylabel('True positive rate')
    ax[0].set_title('ROC curve')
    ax[0].legend(loc='best')

    d_trees = np.arange(dt.get_depth())

    # train_error_lst = []
    # test_error_lst = []
    # for tree in d_trees:
        
    #     # train/test error plot
    #     ax[1].plot(np.arange(dt.get_depth()) + 1, train_error, label='train-error')
    #     ax[1].plot(np.arange(dt.get_depth()) + 1, test_error, label='test-error')
    #     ax[1].set_xlabel('False positive rate')
    #     ax[1].set_ylabel('True positive rate')
    #     ax[1].set_title('ROC curve')
    #     ax[1].legend(loc='best')
    #     plt.show()

    # plt.figure(3)
    # plot_errors()

    ################# ___NEWNEW_______
    # logistic regression model

    # log_turnover = pd.read_csv("../data/turnover.csv")
    # log_turnover["salary"] = log_turnover["salary"].astype('category').cat.reorder_categories(['low', 'medium', 'high']).cat.codes
    # log_turnover.rename(columns={'sales': 'department'}, inplace=True)
    # log_department = pd.get_dummies(log_turnover["department"])

    # log_turnover = log_turnover.drop(["department"], axis=1)


    # X = log_turnover['satisfaction_level'].values
    # y = log_turnover['left'].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # log_model = LogisticRegression()
    # log_model.fit(X_train.reshape(1,-1), y_train)
    # y_hat_prob = log_model.predict_proba(X_test)[:,0] # first column = satisfaction_level
    # print(y_hat_prob)
    # threshold = 0.5
    # y_pred = (y_hat_prob >= threshold).astype(int)
    # print(y_pred)


    # print(log_turnover.head())

    # x_ = np.linspace(0,1,100).reshape(-1,1)
    # sigmoid = log_model.predict_proba(x_)

    # fig = plt.figure(figsize=(8,5))
    # ax = fig.add_subplot(111)
    # ax.scatter(X[:,0], y, c=y, cmap='bwr', edgecolor='')
    # ax.plot(x, sigmoid, 'g--', label='probability of employee turnover')
    # # ax.set_xlim([-0.2,1.201])
    # # ax.set_ylim([-0.2,1.201])
    # ax.set_xlabel('satifaction_level',fontsize=24)
    # ax.set_ylabel('left (1 = Left)',fontsize=24)
    # ax.set_title('Employee Turnover vs. Satisfaction Level',fontsize=24)
    # # ax.legend(fontsize=24)
    # plt.show()

    # exploratory data
    employee_turnvover = turnover.groupby('left').sum()
    employee_turnvover.head()
    # employee_turnvover.columns
    labels = ['IT', 'RandD', 'accounting', 'hr',
        'management', 'marketing', 'product_mng', 'sales', 'support',
        'technical']
    # side by side bar chart
    stayed = employee_turnvover.iloc[0,8:].values
    left = employee_turnvover.iloc[1,8:].values

    fig, ax = plt.subplots(figsize=(12,5))
    width = 0.4
    xlocs = np.arange(len(left))
    ax.bar(xlocs-width, left, width, color='cornflowerblue', label='Left')
    ax.bar(xlocs, stayed, width, color='hotpink', label='Stayed')

    ax.set_xticks(ticks=range(len(left)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    ax.legend(loc='best')
    ax.set_ylabel('Number of Employees')
    ax.set_title('Employee Turnover by Department')
    fig.tight_layout(pad=1)
    # top sales, support, technical 