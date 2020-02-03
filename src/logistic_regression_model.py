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

    ################ ___NEWNEW_______
    logistic regression model

    log_turnover = pd.read_csv("../data/turnover.csv")
    log_turnover["salary"] = log_turnover["salary"].astype('category').cat.reorder_categories(['low', 'medium', 'high']).cat.codes
    log_turnover.rename(columns={'sales': 'department'}, inplace=True)
    log_department = pd.get_dummies(log_turnover["department"])

    log_turnover = log_turnover.drop(["department"], axis=1)


    X = log_turnover['satisfaction_level'].values
    y = log_turnover['left'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_model = LogisticRegression()
    log_model.fit(X_train.reshape(1,-1), y_train)
    y_hat_prob = log_model.predict_proba(X_test)[:,0] # first column = satisfaction_level
    print(y_hat_prob)
    threshold = 0.5
    y_pred = (y_hat_prob >= threshold).astype(int)
    print(y_pred)


    print(log_turnover.head())

    x_ = np.linspace(0,1,100).reshape(-1,1)
    sigmoid = log_model.predict_proba(x_)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], y, c=y, cmap='bwr', edgecolor='')
    ax.plot(x, sigmoid, 'g--', label='probability of employee turnover')
    # ax.set_xlim([-0.2,1.201])
    # ax.set_ylim([-0.2,1.201])
    ax.set_xlabel('satifaction_level',fontsize=24)
    ax.set_ylabel('left (1 = Left)',fontsize=24)
    ax.set_title('Employee Turnover vs. Satisfaction Level',fontsize=24)
    # ax.legend(fontsize=24)
    plt.show()