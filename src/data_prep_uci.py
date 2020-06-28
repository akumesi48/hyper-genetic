import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_cmc = pd.read_csv("data/cmc.data", header=None)
data_cmc[9] = np.where(data_cmc[9] == 1, 0, 1)
data_cmc_label = data_cmc.pop(9)
x_train, x_test, y_train, y_test = train_test_split(data_cmc, data_cmc_label, test_size=0.25)
