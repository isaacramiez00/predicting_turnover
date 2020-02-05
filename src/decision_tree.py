import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

def k_fold_linear():
	''' Returns error for k-fold cross validation. '''
	err_linear, index, num_folds = 0, 0, 5
	kf = KFold(n_splits= num_folds)
	error = np.empty(num_folds)
	linear = LinearRegression()
	for train, test in kf.split(X_train):
		linear.fit(X_train[train], y_train[train])
		pred = linear.predict(X_train[test])
		error[index] = rmse(pred, y_train[test])
		index += 1

	return np.mean(error)

def plot_learning_curve(estimator, label=None):
	''' Plot learning curve with varying training sizes'''
	scores = list()
	train_sizes = np.linspace(10,100,10).astype(int)
	for train_size in train_sizes:
		cv_shuffle = model_selection.ShuffleSplit(train_size=train_size, 
						test_size=200, random_state=0)
		test_error = model_selection.cross_val_score(estimator, X, y, cv=cv_shuffle)
		scores.append(test_error)

	plt.plot(train_sizes, np.mean(scores, axis=1), label=label or estimator.__class__.__name__)
	plt.ylim(0,1)
	plt.title('Learning Curve')
	plt.ylabel('Explained variance on test set (R^2)')
	plt.xlabel('Training test size')
	plt.legend(loc='best')
	plt.show()


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

    # cleaning data
    turnover = pd.read_csv("../data/turnover.csv")
    turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(['low', 'medium', 'high']).cat.codes
    turnover.rename(columns={'sales': 'department'}, inplace=True)
    department = pd.get_dummies(turnover["department"])
    turnover = turnover.drop(["department"], axis=1)

    # changing performance (getting rid of 'satisfaction_level') --- REMEMBER TO CHANGE THIS
    turnover = turnover.drop(['satisfaction_level'], axis=1)

    # # add dummy variables -- not sure if I need it
    # turnover = turnover.join(department)

    # plotting the correlation matrix
    # fig, ax = plt.subplots(figsize=(8,5))
    # ax = sns.heatmap(turnover.corr())
    # plt.show()
    # plt.savefig('correlation_matrix.png')
    # plt.tight_layout()
    # after looking into the correlation matrix, I decided to reduce to 4 features to predict employee turnover

    # # the percentage of leavers
    # turnover['left'].value_counts()/len(turnover)

    # # Decision Tree Classifier
    # features = ['satisfaction_level', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']
    # top_feat = features[:2] # after revisiting
    dt = DecisionTreeClassifier(random_state=42)
    y = turnover.pop('left').values
    X = turnover.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


    # gridsearchCV using eval metrics
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_params_`` and
    # ``gs.best_index_``
    gs = GridSearchCV(dt,
                    param_grid={'min_samples_split': range(2, 403, 10)},
                    scoring=scoring, refit='AUC', return_train_score=True)
    gs.fit(X_train, y_train)
    results = gs.cv_results_
    print(results)


    # plotting gridsearchCV results
    plt.figure(figsize=(8, 8))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
            fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 402)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")  
    plt.grid(False)
    plt.show()



    # dt.fit(X_train, y_train)
    # training_accuracy = dt.score(X_train, y_train)
    # train_error = 1.0 - training_accuracy
    # print(f'training-accuracy: {round(training_accuracy,2)}')
    # print(f'train-error: {round(train_error,2)}')

    # y_pred = dt.predict(X_test)
    # test_accuracy = dt.score(X_test, y_test)
    # test_error = 1.0 - test_accuracy
    # print(f'test-accuracy: {round(test_accuracy, 2)}')
    # print(f'test-error: {round(test_error,2)}')

    # accuracy score plot
    fig, ax = plt.subplots(figsize=(8,5))
    # ax.plot()


    # decision tree plot
    # plt.figure(1)
    # tree.plot_tree(dt.fit(X, y), filled=True)
    # plt.show()

    # feature_importances_
    feat_dict = {k: v for k, v in zip(list(turnover.columns), dt.feature_importances_)} # satisfaction_level and time spend_in_company -> best feat

    # 

    # confusion matrix
    print(confusion_matrix(y_test, y_pred))
    
    # # roc curve summary
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


    # kf = KFold()
    # print(kf.get_n_splits(X_train))
    # for train_index, test_index in kf.split(X):
    #     print(f'TRAIN:, {train_index}, TEST:, {test_index}')
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]


    # estimator = DecisionTreeClassifier()
    # plot_learning_curve(estimator, label='DT')
    # plot_errors()


