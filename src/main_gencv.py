from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# from src.data_preparation import *
from src.data_prep_uci import *

from genetic_selection import GeneticSelectionCV

iris = datasets.load_iris()

# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

X = np.hstack((iris.data, E))
y = iris.target

# estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
estimator = GradientBoostingClassifier(max_features='sqrt',random_state=7840)

selector = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="accuracy",
                              max_features=10,
                              n_population=50,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=40,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=10,
                              caching=True,
                              n_jobs=-1)
selector = selector.fit(X, y)
