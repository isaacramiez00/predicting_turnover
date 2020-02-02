import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

if __name__=='__main__':
    turnover = pd.read_csv("../data/turnover.csv")
    turnover["salary"] = turnover["salary"].astype('category').cat.reorder_categories(['low', 'medium', 'high']).cat.codes
    turnover.rename(columns={'sales': 'department'}, inplace=True)
    department = pd.get_dummies(turnover["department"])
    turnover = turnover.drop(["department"], axis=1)

    # plotting the correlation matrix
    # as seaborn is based on matplotlib, we need to use plt.show() to see the plot
    fig, ax = plt.subplots(figsize=(12,12))
    ax = sns.heatmap(turnover.corr())
    plt.show()
    plt.savefig('correlation_matrix.png')
    plt.tight_layout()

    turnover = turnover.join(department)

    # the percentage of leavers
    turnover['left'].value_counts()/len(turnover)*100