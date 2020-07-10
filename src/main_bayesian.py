from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt import STATUS_OK

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


# def objective_function(params):
#     model = GradientBoostingClassifier(**params, max_features='sqrt')
#     score = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy').mean()
#     return {'auc': score, 'status': STATUS_OK}


param_hyperopt = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 1000, 1)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 15, 1)),
    'min_samples_split': hp.uniform('min_samples_split', 0.01, 1.0),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.5),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
}


def hyperopt(param_space, x_train, y_train, x_test, y_test, cv, num_eval):

    start = time.time()

    def objective_function(params):
        model = GradientBoostingClassifier(**params, max_features='sqrt')
        roc_auc_list = []
        for index in cv:
            model.fit(x_train.iloc[index[0]], y_train.iloc[index[0]])
            # measurement scores
            pred_label = model.predict(x_train.iloc[index[1]])
            pred_prob = model.predict_proba(x_train.iloc[index[1]])
            pred_prob = [x[1] for x in pred_prob]
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train.iloc[index[1]], pred_label)
            roc_auc_list.append(auc(false_positive_rate, true_positive_rate))
        score = round(sum(roc_auc_list)/len(roc_auc_list), 4)
        # score = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy').mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function,
                      param_space,
                      algo=tpe.suggest,
                      max_evals=num_eval,
                      trials=trials,
                      rstate=np.random.RandomState(7840))
    loss = [x['result']['loss'] for x in trials.trials]

    # best_param_values = [x for x in best_param.values()]

    # if best_param_values[0] == 0:
    #     boosting_type = 'gbdt'
    # else:
    #     boosting_type= 'dart'

    clf_best = GradientBoostingClassifier(learning_rate=best_param['learning_rate'],
                                          n_estimators=int(best_param['n_estimators']),
                                          max_depth=int(best_param['max_depth']),
                                          min_samples_split=best_param['min_samples_split'],
                                          min_samples_leaf=best_param['min_samples_leaf'],
                                          subsample=best_param['subsample'],
                                          max_features='sqrt',
                                          random_state=7840,
                                          )

    clf_best.fit(x_train, y_train)
    pred_label = clf_best.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_label)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    print("")
    print("##### Results")
    print("Time elapsed: ", time.time() - start)
    print("Score best parameters: ", max(loss)*-1)
    print("Best parameters: ", best_param)
    print("Test Score: ", clf_best.score(x_test, y_test))
    print(f"AUC SCORE: {roc_auc}")
    print("Parameter combinations evaluated: ", num_eval)

    return trials


# Select Dataset
dataset_name = 'dota'  # titanic, audit, cmc, setap, dota
x_train, x_test, y_train, y_test, index_list = data_selector(dataset_name)

results_hyperopt = hyperopt(param_hyperopt, x_train, y_train, x_test, y_test, index_list, 75)



# # Old code
# start_time = time.time()
#
# tuning = RandomizedSearchCV(estimator=GradientBoostingClassifier(max_features='sqrt',
#                                                                  random_state=7840),
#                             param_distributions=param_test,
#                             scoring='accuracy',
#                             n_iter=10,
#                             n_jobs=-1,
#                             iid=False,
#                             cv=5)
# tuning.fit(x_train, y_train)
# # tuning.grid_scores_, tuning.best_params_, tuning.best_score_
# total_time = (time.time() - start_time)
# print(f"Total time elapse: {total_time}")
#
# validate_score = train_gbm(x_train, y_train, x_test, y_test,
#                            tuning.best_params_['learning_rate'],
#                            tuning.best_params_['n_estimators'],
#                            tuning.best_params_['max_depth'],
#                            tuning.best_params_['min_samples_split'],
#                            tuning.best_params_['min_samples_leaf'],
#                            tuning.best_params_['subsample'])
# print(f"Validate score: {validate_score}")
# print(f"Total time elapse: {total_time}")
# print(tuning.best_params_)
