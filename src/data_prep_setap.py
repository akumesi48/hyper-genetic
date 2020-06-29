import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

data_setap_1 = pd.read_csv("data/setap/setapProcessT1.csv", header=1, skiprows=0)
data_setap_1.rename(columns={data_setap_1.columns[-1]: "label"}, inplace=True)
data_setap_2 = pd.read_csv("data/setap/setapProcessT2.csv", header=1, skiprows=0)
data_setap_2.rename(columns={data_setap_2.columns[-1]: "label"}, inplace=True)
data_setap_3 = pd.read_csv("data/setap/setapProcessT3.csv", header=1, skiprows=0)
data_setap_3.rename(columns={data_setap_3.columns[-1]: "label"}, inplace=True)
data_setap_4 = pd.read_csv("data/setap/setapProcessT4.csv", header=1, skiprows=0)
data_setap_4.rename(columns={data_setap_4.columns[-1]: "label"}, inplace=True)
data_setap_5 = pd.read_csv("data/setap/setapProcessT5.csv", header=1, skiprows=0)
data_setap_5.rename(columns={data_setap_5.columns[-1]: "label"}, inplace=True)
data_setap_6 = pd.read_csv("data/setap/setapProcessT6.csv", header=1, skiprows=0)
data_setap_6.rename(columns={data_setap_6.columns[-1]: "label"}, inplace=True)
data_setap_7 = pd.read_csv("data/setap/setapProcessT7.csv", header=1, skiprows=0)
data_setap_7.rename(columns={data_setap_7.columns[-1]: "label"}, inplace=True)
data_setap_8 = pd.read_csv("data/setap/setapProcessT8.csv", header=1, skiprows=0)
data_setap_8.rename(columns={data_setap_8.columns[-1]: "label"}, inplace=True)
data_setap_9 = pd.read_csv("data/setap/setapProcessT9.csv", header=1, skiprows=0)
data_setap_9.rename(columns={data_setap_9.columns[-1]: "label"}, inplace=True)
data_setap_10 = pd.read_csv("data/setap/setapProcessT10.csv", header=1, skiprows=0)
data_setap_10.rename(columns={data_setap_10.columns[-1]: "label"}, inplace=True)
data_setap_11 = pd.read_csv("data/setap/setapProcessT11.csv", header=1, skiprows=0)
data_setap_11.rename(columns={data_setap_11.columns[-1]: "label"}, inplace=True)
data_setap = pd.concat([data_setap_1, data_setap_2, data_setap_3, data_setap_4,
                        data_setap_5, data_setap_6, data_setap_7, data_setap_8,
                        data_setap_9, data_setap_10, data_setap_11])
data_setap.to_csv("data/setap.csv", index=False)

data_setap['label'] = np.where(data_setap['label'] == 'A', 0, 1)
data_setap_label = data_setap.pop('label')
data_setap_x_train, data_setap_x_test, data_setap_y_train, data_setap_y_test = train_test_split(data_setap,
                                                                                                   data_setap_label,
                                                                                                   test_size=0.25)

