import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, f1_score, brier_score_loss

# get titanic & test csv files as a DataFrame
train = pd.read_csv("data/titanic_train.csv")
print(train.shape)

# Data Cleansing
# Checking for missing data
NAs = pd.concat([train.isnull().sum()], axis=1, keys=['Train'])
NAs[NAs.sum(axis=1) > 0]
# Filling missing Age values with mean
train['Age'] = train['Age'].fillna(train['Age'].mean())
# Filling missing Embarked values with most common value
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train['Pclass'] = train['Pclass'].apply(str)

# Getting Dummies from all other categorical vars
for col in train.dtypes[train.dtypes == 'object'].index:
    for_dummy = train.pop(col)
    train = pd.concat([train, pd.get_dummies(for_dummy, prefix=col)], axis=1)

labels = train.pop('Survived')

# Split train test 75:25
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)


# Train

# Run model function
def train_gbm(trainset_feature, trainset_label, testset_feature, testset_label, learning_rate=0.01, n_estimators=50,
              max_depth=10, min_samples_split=0.5, min_samples_leaf=0.2, max_features=15):
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       max_features=max_features)
    model.fit(trainset_feature, trainset_label)
    pred_label = model.predict(testset_feature)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(testset_label, pred_label)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return roc_auc


model = GradientBoostingClassifier(learning_rate=0.56,
                                   n_estimators=210,
                                   max_depth=5,
                                   min_samples_split=0.6,
                                   min_samples_leaf=0.04,
                                   max_features='sqrt')
model.fit(x_train, y_train)
pred_label = model.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_label)
roc_auc = auc(false_positive_rate, true_positive_rate)
f1_score(y_test, pred_label)
pred = model.predict_proba(x_test)
pred_prob = [x[1] for x in pred]
brier_score_loss(y_test, pred_prob)

# learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
# n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
# max_depths = np.linspace(1, 32, 32, endpoint=True)
# min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
# min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features = list(range(1,train.shape[1]))
# print max_depths
# print min_samples_splits
# print min_samples_leafs
# print max_features
print(train_gbm(x_train, y_train, x_test, y_test))
# print train.shape

# class