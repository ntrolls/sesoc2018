import operator
import math
import random

import numpy

from util import ranking, read_training_data, read_test_data

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 4) # four input variables
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.renameArguments(ARG0='ep')
pset.renameArguments(ARG1='ef')
pset.renameArguments(ARG2='np')
pset.renameArguments(ARG3='nf')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSBFL(individual, training_data):
    # training_data is a list containing 20 instances of the following tuple:
    # (fault_index, spectrum)
    # each of the above tuple corresponds to a fault:
    # fault_index is the 0-based index of the faulty statement, while
    # spectrum is a Python list, containing tuples of (ep, ef, np, nf) for
    # each program statement
    # 
    # for each fault, compile your GP tree into a Python function and
    # compute suspiciousness score for each spectrum tuples
    # then, use ranking(faulty_index, scores) to get the rank of the faulty
    # statement
    #
    # use the average of 20 rankings as the fitness
    
    return numpy.mean(ranks),

training_data = read_training_data();    

toolbox.register("evaluate", evalSBFL, training_data=training_data)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))

def main():

    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.1, 60, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


# Use this function to test your best individual against 
# the reserved test data (i.e. hof[0])
def test(individual):
    test_data = read_test_data()
    formula = toolbox.compile(expr=individual)
    ranks = []
    expenses = []
    for test_set in test_data:
        faulty_index, spectrum = test_set
        susp = []
        for row in spectrum:
            ep, ef, np, nf = row
            susp.append(formula(ep, ef, np, nf))
        rank = ranking(faulty_index, susp)
        ranks.append(rank)
        expenses.append(float(rank) / float(len(spectrum)) * 100)
    print(ranks, numpy.mean(ranks))
    print(expenses, numpy.mean(expenses))

if __name__ == "__main__":
    pop, log, hof = main()
    test(hof[0])