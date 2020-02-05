import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
import pandas as pd
from sklearn import (
    cluster, datasets, 
    decomposition, ensemble, manifold, 
    random_projection, preprocessing)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns



if __name__=='__main__':
    # cleaning data
    turnover = pd.read_csv("../data/turnover.csv")
    turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(['low', 'medium', 'high']).cat.codes
    turnover.rename(columns={'sales': 'department'}, inplace=True)

    department = list(np.array(turnover.department.unique()))
    turnover["department"] = turnover["department"].astype('category').cat.reorder_categories(department).cat.codes
    # print(turnover.department.unique())
    # department = pd.get_dummies(turnover["department"])
    # turnover = turnover.drop(["department"], axis=1)
    # turnover = turnover.join(department)
 


    # plotting the correlation matrix
    # as seaborn is based on matplotlib, we need to use plt.show() to see the plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax = sns.heatmap(turnover.corr())
    plt.show()
    plt.savefig('correlation_matrix.png')
    plt.tight_layout()
    # after looking into the correlation matrix, I decided to reduce to 4 features to predict employee turnover

    # Create classifiers
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier()
    ss = preprocessing.StandardScaler() # pca
    dt = DecisionTreeClassifier(random_state=42)

    y = turnover.pop('left').values
    X = turnover.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # #############################################################################
    # Plot calibration plots

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                    (gnb, 'Naive Bayes'),
                    (svc, 'Support Vector Classification'),
                    (rfc, 'Random Forest'),
                    (dt, 'Decision Tree')]:

        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="%s" % (name, ))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    # plt.show()

    y_pred = rfc.predict(X_test)

    # print(rfc.score(X_test, y_test))
    # print(recall_score(y_test, y_pred, average='binary'))

    features = list(turnover.columns)
    feat_dict = {k: v for k, v in zip(features, rfc.feature_importances_)}
    # print(feat_dict)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    # print(fpr)


    # can create a function that checks different ; y is already set to target value
    features = list(turnover.columns)
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
    # print(results)


    # plotting gridsearchCV results for random forest - probs do other ones too idkkk
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
    # plt.show()
