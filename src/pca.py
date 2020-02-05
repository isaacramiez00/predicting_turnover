from sklearn import (
    cluster, datasets, 
    decomposition, ensemble, manifold, 
    random_projection, preprocessing)
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import statsmodels.api as sm

def scree_plot(ax, pca, n_components_to_plot=8, title=None):
    """Make a scree plot showing the variance explained (i.e. varaince of the projections) for the principal components in a fit sklearn PCA object.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    pca: sklearn.decomposition.PCA object.
      A fit PCA object.
      
    n_components_to_plot: int
      The number of principal components to display in the skree plot.
      
    title: str
      A title for the skree plot.
    """
   
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = np.cumsum(pca.explained_variance_ratio_) # tom udpated
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]), 
                   (ind[i]+0.2, vals[i]+0.005), 
                   va="bottom", 
                   ha="center", 
                   fontsize=12)
    # breakpoint()
    # ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)

def plot_mnist_embedding(ax, X, y, title=None):
    """Plot an embedding of the mnist dataset onto a plane.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
      
    y: numpy.array
      The labels of the datapoints.  Should be digits.
      
    title: str
      A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 str(y[i]),
                 color='r' if str(y[i]) == '0' else 'b',
                 fontdict={'weight': 'bold', 'size': 12})

    ax.set_xticks([]), 
    ax.set_yticks([])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([-0.1,1.1])

    if title is not None:
        ax.set_title(title, fontsize=16)

if __name__=='__main__':
    # clean data
    turnover = pd.read_csv("../data/turnover.csv")
    turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(['low', 'medium', 'high']).cat.codes
    turnover.rename(columns={'sales': 'department'}, inplace=True)
    department = pd.get_dummies(turnover["department"])
    turnover = turnover.drop(["department"], axis=1)
    # turnover = turnover.join(department)

    # splitting data
    y = turnover.pop('left')
    X = turnover # without the department (one-hot encodes), with 6 PC we are capturing 90% variance explained

    ss = preprocessing.StandardScaler()
    X_centered = ss.fit_transform(X)
    # print(X_centered.shape)

    pca = decomposition.PCA(n_components=8)
    X_pca = pca.fit_transform(X_centered)

    # fitting and getting summary statistics
    features = pd.DataFrame(X_pca)

    X = sm.add_constant(features)
    X  =X.reset_index(drop=True)

    est = sm.OLS(y, X).fit()

    print(est.summary())
    # print(X_pca.shape)

    # 90 percent variance
    fig, ax = plt.subplots(figsize=(10, 6))
    scree_plot(ax, pca, title="Scree Plot for Digits Principal Components (CumSum)")
    plt.show()


    pca = decomposition.PCA(n_components=2)
    X_pca = pca.fit_transform(X_centered)
    print(X_pca.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_mnist_embedding(ax, X_pca, y)
    plt.show()



    # pca on my department dataset
    # dep_X = department
    
    # ss = preprocessing.StandardScaler()
    # X_centered = ss.fit_transform(dep_X)
    # print(X_centered.shape)

    # pca = decomposition.PCA(n_components=10)
    # X_pca = pca.fit_transform(X_centered)
    # print(X_pca.shape)

    # # 90 percent variance
    # fig, ax = plt.subplots(figsize=(10, 6))
    # scree_plot(ax, pca, title="Scree Plot for Digits Principal Components (CumSum)")
    # plt.show()

    # pca = decomposition.PCA(n_components=2)
    # X_pca = pca.fit_transform(X_centered)
    # print(X_pca.shape)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # plot_mnist_embedding(ax, X_pca, y)
    # plt.show()

