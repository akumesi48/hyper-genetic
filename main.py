import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from src.hyper_gen import *
from src.data_preparation import *

np.random.seed(7840)

# Configuration for GA parameters
population_size = 8  # number of parents to start
crossover_parent = 4  # number of parents that will mate
no_of_generations = 5  # number of generation that will be created

# initialize the population with randomly generated parameters
populations = init_pop(population_size)
fitness_history = np.empty([no_of_generations + 1, population_size])
genetic_history = np.empty([(no_of_generations + 1) * population_size, populations.shape[1]])
# insert the value of initial parameters in history
genetic_history[0:population_size, :] = populations

for generation in range(no_of_generations):
    print(f"This is number {generation} generation")

    # train the dataset and obtain fitness
    fitness_score = train_populations(population=populations,
                                      trainset_feature=x_train,
                                      trainset_label=y_train,
                                      testset_feature=x_test,
                                      testset_label=y_test)
    fitness_history[generation, :] = fitness_score

    # best score in the current iteration
    print('Best fitness score in the this iteration = {}'.format(np.max(fitness_history[generation, :])))
    # survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be
    # selected
    parents = get_best_population(population=populations,
                                  fitness=fitness_score,
                                  no_of_parent=crossover_parent)

    # mate these parents to create children having parameters from these parents (we are using uniform crossover)
    children = crossover_uniform(parents=parents,
                                 no_of_child=(population_size - parents.shape[0]),
                                 no_of_param=parents.shape[1])

    # add mutation to create genetic diversity
    children_mutated = mutation(children, parents.shape[1])

    '''
    We will create new population, which will contain parents that where selected previously based on the
    fitness score and rest of them  will be children
    '''
    populations[0:parents.shape[0], :] = parents  # fittest parents
    populations[parents.shape[0]:, :] = children_mutated  # children

    # score parent information
    genetic_history[(generation + 1) * population_size: (generation + 1) * population_size + population_size,
    :] = populations

# Best solution from the final iteration
fitness = train_populations(population=populations,
                            trainset_feature=x_train,
                            trainset_label=y_train,
                            testset_feature=x_test,
                            testset_label=y_test)
fitness_history[no_of_generations, :] = fitness
# index of the best solution
bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]
# Best fitness
print("Best fitness is =", fitness[bestFitnessIndex])
# Best parameters
print("Best parameters are:")
print('learning_rate', populations[bestFitnessIndex][0])
print('n_estimators', int(populations[bestFitnessIndex][1]))
print('max_depth', int(populations[bestFitnessIndex][2]))
print('min_samples_split', populations[bestFitnessIndex][3])
print('min_samples_leaf', populations[bestFitnessIndex][4])
print('max_features', int(populations[bestFitnessIndex][5]))
