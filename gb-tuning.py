from extractor import Extractor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

basedata = Extractor.loaddata('data-2-0.5')

# Split the data into test/train
X = basedata[:, :-1]
y = basedata[:, -1]

# RandomSearchCV
# learning_rate = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
# min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features = ['auto', 'log2', None]

# GridSearchCV
learning_rate = [0.05, 0.01, 0.1]
n_estimators = [1200, 1400, 1600]
max_depth = [80, 90, 100, 110]
min_samples_split = [0.05, 0.1, 0.2]
min_samples_leaf = [0.4, 0.5, 0.3]
max_features = [None]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}

gb = GradientBoostingClassifier()
gb_est = GridSearchCV(estimator = gb, param_grid = param_grid, cv = 3, verbose=2, n_jobs = -1)
gb_est.fit(X, y)

print gb_est.best_params_