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
no_of_generation = 4  # number of generation that will be created

# initialize the population with randomly generated parameters
population = init_pop(population_size)
fitness_history = np.empty([no_of_generation + 1, population_size])
genetic_history = np.empty([(no_of_generation + 1) * population_size, population.shape[1]])
# insert the value of initial parameters in history
genetic_history[0:population_size, :] = population

for generation in range(no_of_generation):
    print(f"This is number {generation} generation")

    # train the dataset and obtain fitness
    fitness_score = train_population(population=population, trainset_feature=x_train, trainset_label=y_train,
                                     testset_feature=x_test, testset_label=y_test)
    fitness_history[generation, :] = fitness_score

    # best score in the current iteration
    print('Best fitness score in the this iteration = {}'.format(np.max(fitness_history[generation, :])))
    # survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be
    # selected
    parents = get_best_population(population=population, fitness=fitness_score, no_of_parent=crossover_parent)

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
    population[0:parents.shape[0], :] = parents  # fittest parents
    population[parents.shape[0]:, :] = children_mutated  # children

    # score parent information
    genetic_history[(generation + 1) * population_size: (generation + 1) * population_size + population_size, :] = population
