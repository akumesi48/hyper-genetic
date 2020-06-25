import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import random


def init_pop(no_of_population):
    learning_rates = np.empty([no_of_population, 1])
    n_estimators = np.empty([no_of_population, 1], dtype=np.uint8)
    max_depths = np.empty([no_of_population, 1], dtype=np.uint8)
    min_samples_splits = np.empty([no_of_population, 1])
    min_samples_leafs = np.empty([no_of_population, 1])
    max_features = np.empty([no_of_population, 1], dtype=np.uint8)
    # max_features = np.empty([no_of_population, 1])

    for i in range(no_of_population):
        learning_rates[i] = round(random.uniform(0.01, 1), 2)
        n_estimators[i] = int(random.randrange(10, 500, step=20))
        max_depths[i] = int(random.randrange(1, 15, step=1))
        min_samples_splits[i] = round(random.uniform(0.01, 1.0), 2)
        min_samples_leafs[i] = round(random.uniform(0.01, 0.5), 2)
        max_features[i] = int(random.randrange(2, 1732, step=2))
        # max_features[i] = round(random.uniform(0.01, 1.0), 2)

    population = np.concatenate(
        (learning_rates, n_estimators, max_depths, min_samples_splits, min_samples_leafs, max_features), axis=1)
    return population


# def fitness_f1score(y_true, y_pred):
#     fitness = round((f1_score(y_true, y_pred, average='weighted')), 4)
#     return fitness


# Run model to get fitness function
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


# train the data and find fitness score
def train_populations(population, trainset_feature, trainset_label, testset_feature, testset_label):
    fitness_scores = []
    for i in range(population.shape[0]):
        learning_rate = population[i][0]
        n_estimators = int(population[i][1])
        max_depth = int(population[i][2])
        min_samples_split = population[i][3]
        min_samples_leaf = population[i][4]
        max_features = int(population[i][5])
        # max_features = population[i][5]
        fitness_scores.append(train_gbm(trainset_feature, trainset_label, testset_feature, testset_label,
                                        learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                                        max_features))
    return fitness_scores


# select parents for mating
def get_best_population(population, fitness, no_of_parent):
    top_population = np.empty((no_of_parent, population.shape[1]))  # create an array to store fittest parents

    # find the top best performing parents
    for i in range(no_of_parent):
        pop_id = np.where(fitness == np.amax(fitness))
        top_population[i] = population[pop_id[0][0]]
        fitness[pop_id[0][0]] = -1
    return top_population


'''
Mate these parents to create children having parameters from these parents (we are using uniform crossover method)
'''


def crossover_uniform(parents, no_of_child, no_of_param):
    crossoverPointIndex = np.arange(0, np.uint8(no_of_param), 1, dtype=np.uint8)  # get all the index
    crossoverPointIndex1 = np.random.randint(0, np.uint8(no_of_param), np.uint8(no_of_param / 2))
    crossoverPointIndex2 = np.array(
        list(set(crossoverPointIndex) - set(crossoverPointIndex1)))  # select leftover indexes

    children = np.empty((no_of_child, no_of_param))

    '''
    Create child by choosing parameters from two parents selected using new_parent_selection function. The parameter values
    will be picked from the indexes, which were randomly selected above. 
    '''
    for i in range(no_of_child):
        # find parent 1 index
        parent1_index = i % parents.shape[0]
        # find parent 2 index
        parent2_index = (i + 1) % parents.shape[0]
        # insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex1] = parents[parent1_index, crossoverPointIndex1]
        # insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex2] = parents[parent2_index, crossoverPointIndex2]

    return children


def mutation(crossover, no_of_param):
    # Define minimum and maximum values allowed for each parameter
    constraint_cap = np.zeros((no_of_param, 2))

    constraint_cap[0:] = [0.01, 1.0]  # min/max learning rate
    constraint_cap[1, :] = [10, 2000]  # min/max n_estimator
    constraint_cap[2, :] = [1, 15]  # min/max depth
    constraint_cap[3, :] = [0.01, 1.0]  # min/max min_samples_split
    constraint_cap[4, :] = [0.01, 0.5]  # min/max min_samples_leaf
    constraint_cap[5, :] = [2, 1732]  # min/max max_features
    # constraint_cap[5, :] = [0.01, 1.0]  # min/max max_features

    # learning_rates[i] = round(random.uniform(0.01, 1), 2)
    # n_estimators[i] = int(random.randrange(10, 500, step=20))
    # max_depths[i] = int(random.randrange(1, 32, step=1))
    # min_samples_splits[i] = round(random.uniform(0.01, 1.0), 2)
    # min_samples_leafs[i] = round(random.uniform(0.01, 10.0), 2)
    # max_features[i] = int(random.randrange(2, 1732, step=2))

    # Mutation changes a single gene in each offspring randomly.
    mutationValue = 0
    parameterSelect = np.random.randint(0, 6, 1)
    print(parameterSelect)
    if parameterSelect == 0:  # learning_rate
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 1:  # n_estimators
        mutationValue = np.random.randint(-200, 200, 1)
    if parameterSelect == 2:  # max_depth
        mutationValue = np.random.randint(-5, 15, 1)
    if parameterSelect == 3:  # min_samples_split
        mutationValue = round(np.random.uniform(-0.5, 1.0), 2)
    if parameterSelect == 4:  # min_samples_leaf
        mutationValue = round(np.random.uniform(-0.1, 0.5), 2)
    if parameterSelect == 5:  # max_features
        mutationValue = np.random.randint(-1000, 2000, 1)
        # mutationValue = round(np.random.uniform(-0.5, 1.0), 2)

    # indtroduce mutation by changing one parameter, and set to max or min if it goes out of range
    for idx in range(crossover.shape[0]):
        crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue
        if (crossover[idx, parameterSelect] > constraint_cap[parameterSelect, 1]):
            crossover[idx, parameterSelect] = constraint_cap[parameterSelect, 1]
        if (crossover[idx, parameterSelect] < constraint_cap[parameterSelect, 0]):
            crossover[idx, parameterSelect] = constraint_cap[parameterSelect, 0]

    return crossover
