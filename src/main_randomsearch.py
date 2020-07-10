from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np

# from src.data_preparation import *
from src.data_prep_uci import *


# Run model function
def train_gbm(trainset_feature, trainset_label, testset_feature, testset_label, learning_rate=0.01, n_estimators=50,
              max_depth=10, min_samples_split=0.5, min_samples_leaf=0.2, subsample=1):
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       subsample=subsample, max_features='sqrt')
    model.fit(trainset_feature, trainset_label)
    pred_label = model.predict(testset_feature)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(testset_label, pred_label)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return roc_auc


# Select Dataset
dataset_name = 'titanic'  # titanic, audit, cmc, setap, dota
x_train, x_test, y_train, y_test, index_list = data_selector(dataset_name)

param_test = {'learning_rate': [round(x, 4) for x in list(np.linspace(0.01, 1, 100))],
              'n_estimators': list(range(10, 1000)),
              'max_depth': list(range(1, 15)),
              'min_samples_split': [round(x, 4) for x in list(np.linspace(0.01, 1, 100))],
              'min_samples_leaf': [round(x, 4) for x in list(np.linspace(0.01, 0.5, 100))],
              'subsample': [round(x, 4) for x in list(np.linspace(0.5, 1, 6))]}

start_time = time.time()

tuning = RandomizedSearchCV(estimator=GradientBoostingClassifier(max_features='sqrt',
                                                                 random_state=7840),
                            param_distributions=param_test,
                            scoring='accuracy',
                            n_iter=10,
                            n_jobs=-1,
                            iid=False,
                            cv=5)
tuning.fit(x_train, y_train)
# tuning.grid_scores_, tuning.best_params_, tuning.best_score_
total_time = (time.time() - start_time)
print(f"Total time elapse: {total_time}")

validate_score = train_gbm(x_train, y_train, x_test, y_test,
                           tuning.best_params_['learning_rate'],
                           tuning.best_params_['n_estimators'],
                           tuning.best_params_['max_depth'],
                           tuning.best_params_['min_samples_split'],
                           tuning.best_params_['min_samples_leaf'],
                           tuning.best_params_['subsample'])
print(f"Validate score: {validate_score}")
print(f"Total time elapse: {total_time}")
print(tuning.best_params_)

