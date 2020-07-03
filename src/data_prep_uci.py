import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold


def cv_index(n_fold, feature, label):
    skf = KFold(n_fold)
    index_list = []
    for i, j in skf.split(feature, label):
        index_list.append((i, j))
    return index_list


def data_selector(data_name):
    if data_name == 'cmc':
        return x_train_cmc, x_test_cmc, y_train_cmc, y_test_cmc, index_cmc
    elif data_name == 'setap':
        return x_train_setap, x_test_setap, y_train_setap, y_test_setap, index_setap
    elif data_name == 'audit':
        return x_train_audit, x_test_audit, y_train_audit, y_test_audit, index_audit
    elif data_name == 'titanic':
        return x_train_tt, x_test_tt, y_train_tt, y_test_tt, index_tt
    elif data_name == 'dota':
        return x_train_dota, x_test_dota, y_train_dota, y_test_dota, index_dota


no_of_folds = 3

# Dataset cmc
data_cmc = pd.read_csv("data/cmc.data", header=None)
data_cmc[9] = np.where(data_cmc[9] == 1, 0, 1)
data_cmc_label = data_cmc.pop(9)
x_train_cmc, x_test_cmc, y_train_cmc, y_test_cmc = train_test_split(data_cmc, data_cmc_label, test_size=0.25)
index_cmc = cv_index(no_of_folds, x_train_cmc, y_train_cmc)

# Dataset SETAP
data_setap = pd.read_csv("data/setap.csv")
data_setap['label'] = np.where(data_setap['label'] == 'A', 0, 1)
data_setap_label = data_setap.pop('label')
x_train_setap, x_test_setap, y_train_setap, y_test_setap = train_test_split(data_setap,
                                                                            data_setap_label,
                                                                            test_size=0.25)
index_setap = cv_index(no_of_folds, x_train_setap, y_train_setap)

# Dataset audit
data_audit = pd.read_csv("data/audit_risk.csv")
data_audit['LOCATION_ID'] = pd.to_numeric(data_audit['LOCATION_ID'], errors='coerce')
data_audit['LOCATION_ID'] = data_audit['LOCATION_ID'].fillna(data_audit['LOCATION_ID'].mode()[0])
data_audit['Money_Value'] = data_audit['Money_Value'].fillna(data_audit['Money_Value'].mean())
data_audit_label = data_audit.pop('Risk')
x_train_audit, x_test_audit, y_train_audit, y_test_audit = train_test_split(data_audit,
                                                                            data_audit_label,
                                                                            test_size=0.25)
index_audit = cv_index(no_of_folds, x_train_audit, y_train_audit)

# Dataset titanic
data_tt = pd.read_csv("data/titanic_train.csv")
data_tt['Age'] = data_tt['Age'].fillna(data_tt['Age'].mean())
data_tt['Embarked'] = data_tt['Embarked'].fillna(data_tt['Embarked'].mode()[0])
data_tt['Pclass'] = data_tt['Pclass'].apply(str)
for col in data_tt.dtypes[data_tt.dtypes == 'object'].index:
    for_dummy = data_tt.pop(col)
    data_tt = pd.concat([data_tt, pd.get_dummies(for_dummy, prefix=col)], axis=1)
data_tt_labels = data_tt.pop('Survived')
x_train_tt, x_test_tt, y_train_tt, y_test_tt = train_test_split(data_tt, data_tt_labels, test_size=0.25)
index_tt = cv_index(no_of_folds, x_train_tt, y_train_tt)

# Dataset DotA2
x_train_dota = pd.read_csv("../dota2Dataset/dota2Train.csv", header=None)
x_train_dota[0] = np.where(x_train_dota[0] == 1, 1, 0)
y_train_dota = x_train_dota.pop(0)
x_test_dota = pd.read_csv("../dota2Dataset/dota2Test.csv", header=None)
x_test_dota[0] = np.where(x_test_dota[0] == 1, 1, 0)
y_test_dota = x_test_dota.pop(0)
index_dota = cv_index(no_of_folds, x_train_dota, y_train_dota)

# for train_index, test_index in skf.split(x_train, y_train):
#     train_feature, test_feature = x_train.iloc[train_index], x_train.iloc[test_index]
#     train_label, test_label = y_train.iloc[train_index], y_train.iloc[test_index]
#     print(train_gbm(train_feature, train_label, test_feature, test_label))

# skf = KFold(5)
# train_index = []
# test_index = []
# index_list = []
# for i, j in skf.split(x_train_cmc, y_train_cmc):
#     index_list.append((i, j))
