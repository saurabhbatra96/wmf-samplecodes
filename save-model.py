import numpy as np
import pickle
from extractor import Extractor
from sklearn.ensemble import RandomForestClassifier

basedata = Extractor.loaddata('data-2-0.5')

X = basedata[:, :-1]
y = basedata[:, -1]

clf = RandomForestClassifier()
clf.fit(X, y)

fname = 'models/rfc-default.sav'
pickle.dump(clf, open(fname, 'wb'))
