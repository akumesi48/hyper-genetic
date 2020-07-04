import logging
import time
from datetime import datetime

# from src.data_preparation import *
from src.data_prep_uci import *
from src.hyper_gen import *


# Select Dataset
dataset_name = 'titanic'  # titanic, audit, cmc, setap, dota
x_train, x_test, y_train, y_test, index_list = data_selector(dataset_name)

# Configuration for GA parameters
population_size = 20
no_of_generations = 40
crossover_parent = 4
crossover_ratio = 0.75
mutation_prob = 0.03
mutation_rate = 0.5
stopping_criteria = 3


file_date = datetime.today()
logger = logging.getLogger('hyper-gen')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('logs/data_{}_{}.log'.format(dataset_name, file_date.strftime("%Y%m%d")))
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

# np.random.seed(7840)
random.seed(7840)
print(f"Start simulation at {file_date}")
print(f"data set = {dataset_name}")

start_time = time.time()
logger.warning(f"Start simulation at {file_date}")
logger.debug(f"set max_feature = 'sqrt' and use brier score with dynamic pc pm")
logger.debug(f"data set = {dataset_name}")
logger.info(f"population size = {population_size}")
logger.info(f"number of generation = {no_of_generations}")
logger.info(f"best elite select = {crossover_parent}")
logger.info(f"crossover ratio = {crossover_ratio}")
logger.info(f"mutation probability and rate = {mutation_prob}, {mutation_rate}")
generations = [Generation()]
generations[0].init_pop(population_size)
stopping_track = (-1, 0)
for gen_no in range(no_of_generations-1):
    print(f"Training generation number {gen_no}")
    logger.info(f"Training generation number {gen_no}")
    # generations[gen_no].train_populations(x_train, y_train, x_test, y_test)
    generations[gen_no].train_populations(x_train, y_train, index_list=index_list)
    generations[gen_no].select_survived_pop(crossover_parent)
    gen_score = generations[gen_no].survived_populations[0].get_score()
    print(f"Best score of generation {gen_no} : score={gen_score[0]}, auc={gen_score[1]}, brier={gen_score[3]}")
    logger.info(f"Best score of generation {gen_no} : score={gen_score[0]}, auc={gen_score[1]}, brier={gen_score[3]}")
    stopping_track = stopping_update(stopping_track, gen_score[0])
    if stopping_track[1] == stopping_criteria:
        print("Stopping criteria reached.")
        logger.debug("Stopping criteria reached.")
        break
    # generations[gen_no].survived_populations[0].explain(logger)
    crossover_ratio = (gen_no+1)/no_of_generations
    child = generations[gen_no].cross_over(crossover_ratio)
    generations.append(Generation(child))
    mutation_prob = 1 - (gen_no+1)/no_of_generations  # dynamic mutation
    print(f"mutation  with probability = {mutation_prob}")
    generations[gen_no+1].mutation(mutation_prob)
    # generations[gen_no+1].mutation(mutation_prob, mutation_rate)
    generations[gen_no+1].add_population(generations[gen_no].survived_populations)
if stopping_track[1] != stopping_criteria:
    print(f"Training generation number {no_of_generations-1}")
    logger.info(f"Training generation number {no_of_generations-1}")
    # generations[no_of_generations-1].train_populations(x_train, y_train, x_test, y_test)
    generations[no_of_generations-1].train_populations(x_train, y_train, index_list=index_list)
    generations[no_of_generations-1].select_survived_pop(crossover_parent)
gen_score = generations[-1].survived_populations[0].get_score()
logger.info(f"Best score of final generation : score={gen_score[0]}, auc={gen_score[1]}, brier={gen_score[3]}")
generations[-1].survived_populations[0].explain(logger)

print(f"##### Best individual: ")
print(f"Best score of final generation : score={gen_score[0]}, auc={gen_score[1]}, brier={gen_score[3]}")
generations[-1].survived_populations[0].explain()

evaluator = generations[-1].survived_populations[0]
evaluator.train_model(x_train, y_train, x_test, y_test)
print(f"#### Evaluation score: auc = {evaluator.auc}, brier = {evaluator.brier_score}")
logger.info(f"#### Evaluation score: auc = {evaluator.auc}, brier = {evaluator.brier_score}")

total_time = (time.time() - start_time)
print(f"Total time elapse: {total_time}")
logger.info(f"Total elapse time: {total_time}")


# # All elite population
# elite_population = []
# elite_score = []
# for gen in generations:
#     elite_population += [x for x in gen.survived_populations]
#     elite_score += [x.score for x in gen.survived_populations]
# elite_population = Generation(elite_population)
# elite_population.fitness_scores = elite_score
# for i in range(len(elite_population.populations)):
#     elite_population.populations[i].score = elite_score[i]
# elite_population.select_survived_pop(10)
