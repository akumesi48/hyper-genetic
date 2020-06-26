import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from src.hyper_gen import *
from src.data_preparation import *
import time

# np.random.seed(7840)
# random.seed(7840)

# Configuration for GA parameters
population_size = 20
crossover_parent = 4
no_of_generations = 10

start_time = time.time()

generations = [Generation()]
generations[0].init_pop(population_size)
for gen_no in range(no_of_generations-1):
    print(f"Training generation number {gen_no}")
    generations[gen_no].train_populations(x_train, y_train, x_test, y_test)
    generations[gen_no].select_survived_pop(crossover_parent)
    print(f"Best score of generation {gen_no} : {generations[gen_no].survived_populations[0].score}")
    # generations[gen_no].survived_populations[0].explain()
    child = generations[gen_no].cross_over(0.8)
    generations.append(Generation(child))
    generations[gen_no+1].mutation(0.3, 0.5)
    generations[gen_no+1].add_population(generations[gen_no].survived_populations)
print(f"Training generation number {no_of_generations-1}")
generations[-1].train_populations(x_train, y_train, x_test, y_test)
generations[-1].select_survived_pop(crossover_parent)
generations[-1].survived_populations[0].explain()

total_time = (time.time() - start_time)
print(f"Total time elapse: {total_time}")
