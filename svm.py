import matplotlib.pyplot as plt
from extractor import Extractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn import svm

basedata = Extractor.loaddata('svmdata0.5')

# Split the data into test/train
X = basedata[:, :-1]
y = basedata[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Classification
clf = svm.SVC()
clf.fit(X_train, y_train)
test_score = clf.score(X_test, y_test)
y_score = clf.decision_function(X_test)


# Get the P-R curve - http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
average_precision = average_precision_score(y_test, y_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}, Accuracy scores = {1:0.02f}, Fnf split = {2:0.02f}'.format(
          average_precision, test_score, 0.5))
plt.show()