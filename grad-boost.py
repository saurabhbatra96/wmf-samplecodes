import matplotlib.pyplot as plt
import numpy as np
from extractor import Extractor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier


basedata = Extractor.loaddata('data-1-0.5')

# Split the data into test/train
X = basedata[:, :-1]
y = basedata[:, -1]

# Use cross-validation
y_tests = np.array([0.0])
y_scores = np.array([0.0])

skf = StratifiedKFold(n_splits=4, shuffle=True)
for train_index, test_index in skf.split(X, y):
	X_train, X_test = np.array(X[train_index]), np.array(X[test_index])
	y_train, y_test = np.array(y[train_index]), np.array(y[test_index])

	# Classification
	clf = GradientBoostingClassifier()
	clf.fit(X_train, y_train)
	y_score = clf.predict_proba(X_test)[:,1]

	y_scores = np.hstack((y_scores, y_score))
	y_tests = np.hstack((y_tests, y_test))

y_tests = y_tests[1:]
y_scores = y_scores[1:]

average_precision = average_precision_score(y_tests, y_scores)
print('Average precision score: {0:0.2f}'.format(average_precision))

precision, recall, thresholds = precision_recall_curve(y_tests, y_scores)

graph = plt.figure()
axes = graph.add_axes([0.1,0.1,0.75,0.75])
axes.set_title("Gradient Boost Default Precision vs Recall")
axes.set_xlabel("Threshold")

axes.plot(thresholds, recall[:-1], "#00CFFF", label="Recall")
axes.plot(thresholds, precision[:-1], "#879908", label="Precision")

axes.legend(loc="center left")
# graph.savefig("plots/gb-default.png")

