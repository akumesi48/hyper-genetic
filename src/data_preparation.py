import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


# get titanic & test csv files as a DataFrame
train = pd.read_csv("data/titanic_train.csv")
# print(train.shape)

# Data Cleansing
# Checking for missing data
NAs = pd.concat([train.isnull().sum()], axis=1, keys=['Train'])
# NAs[NAs.sum(axis=1) > 0]
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

x_train_tt, x_test_tt, y_train_tt, y_test_tt = train_test_split(train, labels, test_size=0.25)

skf = KFold(5)
train_index = []
test_index = []
for i, j in skf.split(x_train, y_train):
    train_index.append(i)
    test_index.append(j)
index_list = (train_index, test_index)