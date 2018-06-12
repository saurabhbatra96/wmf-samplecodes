import matplotlib.pyplot as plt
from extractor import Extractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn import svm

basedata = Extractor.loaddata('data-1-0.5')

# Split the data into test/train
X = basedata[:, :-1]
y = basedata[:, -1]
# 70% vs 30% train-test split. Stratify ensures equal distribution of +ve and -ve samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Classification
clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)
test_score = clf.score(X_test, y_test)
y_score = clf.predict_proba(X_test)[:,1]


# Get the P-R curve - http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
# https://github.com/adamwight/thresholds_diagrams/blob/master/Thresholds%20diagrams.ipynb
average_precision = average_precision_score(y_test, y_score)
print('Average precision score: {0:0.2f}'.format(average_precision))

precision, recall, thresholds = precision_recall_curve(y_test, y_score)

graph = plt.figure()
axes = graph.add_axes([0.1,0.1,0.75,0.75])
axes.set_title("SVM Default Precision vs Recall")
axes.set_xlabel("Threshold")

axes.plot(thresholds, recall[:-1], "#00CFFF", label="Recall")
axes.plot(thresholds, precision[:-1], "#879908", label="Precision")
# axes.plot(recall, precision, "#00CFFF", label="Prec vs Recall")

axes.legend(loc="center left")
graph.savefig("plots/svm-default.png")

