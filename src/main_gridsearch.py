from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# from src.data_preparation import *
from src.data_prep_uci import *

start_time = time.time()
baseline = GradientBoostingClassifier(learning_rate=0.1,
                                      n_estimators=100,
                                      max_depth=3,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      subsample=1,
                                      max_features='sqrt',
                                      random_state=10)
baseline.fit(x_train, y_train)
predictors = list(x_train)
feat_imp = pd.Series(baseline.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importance of Features')
plt.ylabel('Feature Importance Score')
print('Accuracy of the GBM on test set: {:.3f}'.format(baseline.score(x_test, y_test)))
pred = baseline.predict(x_test)
print(classification_report(y_test, pred))

total_time = (time.time() - start_time)
print(f"Total time elapse: {total_time}")

# self.learning_rate = round(random.uniform(0.01, 1), 2)
# self.n_estimators = int(random.randrange(10, 500, step=20))
# self.max_depth = int(random.randrange(1, 15, step=1))
# self.min_samples_split = round(random.uniform(0.01, 1.0), 2)
# self.min_samples_leaf = round(random.uniform(0.01, 0.5), 2)
# self.subsample = round(random.uniform(0.7, 1), 2)
# param_test = {'learning_rate': list(np.arange(0.01, 1.01, 0.01)),
#               'n_estimators': list(np.arange(10, 501, 20)),
#               'max_depth': list(np.arange(1, 16, 1)),
#               'min_samples_split': list(np.arange(0.01, 1.01, 0.01)),
#               'min_samples_leaf': list(np.arange(0.01, 0.51, 0.01)),
#               'subsample': list(np.arange(0.7, 1.01, 0.01))}

# learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
# n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
# max_depths = np.linspace(1, 32, 32, endpoint=True)
# min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
# min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

# param_test = {'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
#               'n_estimators': [10, 20, 40, 80, 100, 300, 600, 1000],
#               'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#               'min_samples_split': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
#               'min_samples_leaf': [0.01, 0.05, 0.1, 0.25, 0.5],
#               'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}


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
dataset_name = 'dota'  # titanic, audit, cmc, setap, dota
x_train, x_test, y_train, y_test, index_list = data_selector(dataset_name)

param_test = {'learning_rate': [0.5, 0.1, 0.05, 0.01],
              'n_estimators': [10, 50, 100, 500],
              'max_depth': [1, 2, 3, 4, 5, 6],
              'min_samples_split': [0.01, 0.05, 0.1, 0.5, 1.0],
              'min_samples_leaf': [0.01, 0.05, 0.1, 0.5],
              'subsample': [0.7, 0.8, 0.8, 0.9]}

start_time = time.time()
# p_test4 = {'min_samples_split': [2, 4, 6, 8, 10, 20, 40, 60, 100],
#            'min_samples_leaf': [1, 3, 5, 7, 9]}

tuning = GridSearchCV(estimator=GradientBoostingClassifier(max_features='sqrt',
                                                           random_state=7840),
                      param_grid=param_test,
                      scoring='accuracy',
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

