import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, f1_score, brier_score_loss
import random
import time
import logging


class Individual:
    def __init__(self, parameters=None):
        if parameters is None:
            self.learning_rate = round(random.uniform(0.01, 1), 2)
            self.n_estimators = int(random.randrange(10, 1000, step=20))
            self.max_depth = int(random.randrange(1, 15, step=1))
            self.min_samples_split = round(random.uniform(0.01, 1.0), 2)
            self.min_samples_leaf = round(random.uniform(0.01, 0.5), 2)
            self.subsample = round(random.uniform(0.7, 1), 2)
        else:
            self.learning_rate = parameters[0]
            self.n_estimators = parameters[1]
            self.max_depth = parameters[2]
            self.min_samples_split = parameters[3]
            self.min_samples_leaf = parameters[4]
            self.subsample = parameters[5]
        self.score = -1
        self.auc = -1
        self.f1_score = -1
        self.brier_score = -1
        self.time = -1

    def train_model(self, train_feature, train_label, test_feature, test_label):
        start_time = time.time()
        model = GradientBoostingClassifier(learning_rate=self.learning_rate,
                                           n_estimators=self.n_estimators,
                                           max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           subsample=self.subsample,
                                           max_features='sqrt',
                                           random_state=7840)
        model.fit(train_feature, train_label)
        total_time = (time.time() - start_time)
        self.time = total_time

        # measurement scores
        pred_label = model.predict(test_feature)
        pred_prob = model.predict_proba(test_feature)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_label, pred_label)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        pred_prob = [x[1] for x in pred_prob]
        self.auc = round(roc_auc, 4)
        self.f1_score = round(f1_score(test_label, pred_label), 4)
        self.brier_score = round(brier_score_loss(test_label, pred_prob), 4)

    def train_model_cv(self, df_feature, df_label, index_list):
        start_time = time.time()
        model = GradientBoostingClassifier(learning_rate=self.learning_rate,
                                           n_estimators=self.n_estimators,
                                           max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           subsample=self.subsample,
                                           max_features='sqrt',
                                           random_state=7840)
        roc_auc_list = []
        f1_score_list = []
        brier_score_list = []
        for index in index_list:
            model.fit(df_feature.iloc[index[0]], df_label.iloc[index[0]])
            # measurement scores
            pred_label = model.predict(df_feature.iloc[index[1]])
            pred_prob = model.predict_proba(df_feature.iloc[index[1]])
            pred_prob = [x[1] for x in pred_prob]
            false_positive_rate, true_positive_rate, thresholds = roc_curve(df_label.iloc[index[1]], pred_label)
            roc_auc_list.append(auc(false_positive_rate, true_positive_rate))
            f1_score_list.append(f1_score(df_label.iloc[index[1]], pred_label))
            brier_score_list.append(brier_score_loss(df_label.iloc[index[1]], pred_prob))

        total_time = (time.time() - start_time)
        self.time = total_time

        self.auc = round(sum(roc_auc_list)/len(roc_auc_list), 4)
        self.f1_score = round(sum(f1_score_list)/len(f1_score_list), 4)
        self.brier_score = round(sum(brier_score_list)/len(brier_score_list), 4)

    def fitness_func(self):
        fitness_score = (1 - self.brier_score)*self.auc  # determine fitness function here
        self.score = fitness_score
        return fitness_score

    def get_score(self):
        return self.score, self.auc, self.f1_score, self.brier_score

    def get_param(self):
        return ([self.learning_rate,
                 self.n_estimators,
                 self.max_depth,
                 self.min_samples_split,
                 self.min_samples_leaf,
                 self.subsample])

    def explain(self, logger=None):
        if logger is None:
            print(f"Individual Score: {self.score}")
            print("Parameters:")
            print(f"learning_rate       = {self.learning_rate}")
            print(f"n_estimators        = {self.n_estimators}")
            print(f"max_depth           = {self.max_depth}")
            print(f"min_samples_split   = {self.min_samples_split}")
            print(f"min_samples_leaf    = {self.min_samples_leaf}")
            print(f"subsample           = {self.subsample}")
        else:
            logger.info(f"Individual Score: {self.score}")
            logger.info("Parameters:")
            logger.info(f"learning_rate       = {self.learning_rate}")
            logger.info(f"n_estimators        = {self.n_estimators}")
            logger.info(f"max_depth           = {self.max_depth}")
            logger.info(f"min_samples_split   = {self.min_samples_split}")
            logger.info(f"min_samples_leaf    = {self.min_samples_leaf}")
            logger.info(f"subsample           = {self.subsample}")


class Generation:
    def __init__(self, population_list=None):
        if population_list is None:
            self.populations = []
        else:
            self.populations = population_list
        self.survived_populations = []
        self.fitness_scores = []

    def init_pop(self, population_size):
        for i in range(population_size):
            self.populations.append(Individual())

    def train_populations(self, train_feature, train_label, test_feature=None, test_label=None, index_list=None):
        for individual in self.populations:
            if index_list is None:
                individual.train_model(train_feature, train_label, test_feature, test_label)
            else:
                individual.train_model_cv(train_feature, train_label, index_list)
            score = individual.fitness_func()
            self.fitness_scores.append(score)

    def select_survived_pop(self, survive_individual):
        self.survived_populations = []
        score_rank = list(enumerate(self.fitness_scores))
        score_rank.sort(reverse=True, key=lambda x: x[1])
        for rank in score_rank[:survive_individual]:
            self.survived_populations.append(self.populations[rank[0]])

    def add_population(self, new_populations):
        self.populations += new_populations

    def cross_over(self, cv_ratio=1.0):
        no_of_child = len(self.populations) - len(self.survived_populations)
        child_list = []
        parent_index = [0, 1]
        # check case for single survived population
        if len(self.survived_populations) > 1:
            for i in range(round(cv_ratio*no_of_child)):
                if len(self.survived_populations) > 2:
                    # random select two candidate from survived population
                    candidates_list = range(len(self.populations))
                    parent_index[0] = random.choice(candidates_list)
                    parent_index[1] = random.choice(candidates_list)
                    while parent_index[0] == parent_index[1]:
                        parent_index[1] = random.choice(candidates_list)
                # crossover to breed new population
                param_index = [random.randint(0, 1) for index in range(6)]
                child = Individual([self.populations[parent_index[param_index[0]]].learning_rate,
                                    self.populations[parent_index[param_index[1]]].n_estimators,
                                    self.populations[parent_index[param_index[2]]].max_depth,
                                    self.populations[parent_index[param_index[3]]].min_samples_split,
                                    self.populations[parent_index[param_index[4]]].min_samples_leaf,
                                    self.populations[parent_index[param_index[5]]].subsample])
                child_list.append(child)
        # fill the rest with immigrant population
        for i in range(no_of_child - (round(cv_ratio*no_of_child))):
            child_list.append(Individual())
        return child_list

    def mutation(self, prob_of_mutation, mutation_rate=None):
        no_of_parameters = 6
        if mutation_rate is not None:
            no_of_mutation = round(mutation_rate*no_of_parameters)
            for individual in self.populations:
                if random.uniform(0, 1) < prob_of_mutation:
                    param_index = [random.randint(0, 6) for index in range(no_of_mutation)]
                    if 0 in param_index:
                        individual.learning_rate = round(random.uniform(0.01, 1), 2)
                    if 1 in param_index:
                        individual.n_estimators = int(random.randrange(10, 500, step=20))
                    if 2 in param_index:
                        individual.max_depth = int(random.randrange(1, 15, step=1))
                    if 3 in param_index:
                        individual.min_samples_split = round(random.uniform(0.01, 1.0), 2)
                    if 4 in param_index:
                        individual.min_samples_leaf = round(random.uniform(0.01, 0.5), 2)
                    if 5 in param_index:
                        individual.subsample = round(random.uniform(0.7, 1), 2)
        else:
            for individual in self.populations:
                if random.uniform(0, 1) < prob_of_mutation:
                    individual.learning_rate = round(random.uniform(0.01, 1), 2)
                if random.uniform(0, 1) < prob_of_mutation:
                    individual.n_estimators = int(random.randrange(10, 500, step=20))
                if random.uniform(0, 1) < prob_of_mutation:
                    individual.max_depth = int(random.randrange(1, 15, step=1))
                if random.uniform(0, 1) < prob_of_mutation:
                    individual.min_samples_split = round(random.uniform(0.01, 1.0), 2)
                if random.uniform(0, 1) < prob_of_mutation:
                    individual.min_samples_leaf = round(random.uniform(0.01, 0.5), 2)
                if random.uniform(0, 1) < prob_of_mutation:
                    individual.subsample = round(random.uniform(0.7, 1), 2)


def stopping_update(tracker, score):
    res_score = tracker[0]
    res_counter = tracker[1]
    if score > tracker[0]:
        res_counter = 0
        res_score = score
    else:
        res_counter += 1
    return (res_score, res_counter)


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

    # introduce mutation by changing one parameter, and set to max or min if it goes out of range
    for idx in range(crossover.shape[0]):
        crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue
        if (crossover[idx, parameterSelect] > constraint_cap[parameterSelect, 1]):
            crossover[idx, parameterSelect] = constraint_cap[parameterSelect, 1]
        if (crossover[idx, parameterSelect] < constraint_cap[parameterSelect, 0]):
            crossover[idx, parameterSelect] = constraint_cap[parameterSelect, 0]

    return crossover
