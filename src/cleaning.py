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
    print(f'training-accuracy: {round(training_accuracy,2)}')

    y_pred = dt.predict(X_test)
    test_accuracy = dt.score(X_test, y_test)
    print(f'test-accuracy: {round(test_accuracy, 2)}')

    plt.figure(1)
    tree.plot_tree(dt.fit(X, y), filled=True)
    plt.show()

    # feature_importances_
    feat_dict = {k: v for k, v in zip(features, dt.feature_importances_)} # satisfaction_level and time spend_in_company -> best feat

    # # confusion matrix
    # print(confusion_matrix(y_test, y_pred))
    
    # # roc curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    # print(auc(fpr, tpr))

    # plt.figure(2)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr)
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # # plt.legend(loc='best')
    # plt.show()

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