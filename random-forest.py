import matplotlib.pyplot as plt
from extractor import Extractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier


basedata = Extractor.loaddata('rfdata0.5')

# Split the data into test/train
X = basedata[:, :-1]
y = basedata[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Classification
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
test_score = clf.score(X_test, y_test)
print test_score