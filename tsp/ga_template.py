import sys
import random
import numpy
from evaluate import evaluate

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

IND_SIZE = 127

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

# Initialize Toolbox

toolbox = base.Toolbox()
# random permutation
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
# individual with list representation
toolbox.register("individual", tools.initIterate, creator.Individual, 
    toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# fitness evaluation uses given utility function
def evalTour(ind):
	return evaluate(ind),

# crossover
def crossover(ind1, ind2):
    assert(len(ind1) == len(ind2))
    # cross-over ind1 and ind2 to generate child_a and child_b
    # tip: you can do the following to clone parent
    # child_a = toolbox.clone(ind1)

    return child_a, child_b

    
def mutate(individual):
    # mutate individual before returning it

    return individual,

toolbox.register("evaluate", evalTour)
toolbox.register("mate",crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # population size 50
    pop = toolbox.population(50)

    # hall of fame: archives the best solution during evolution
    hof = tools.ParetoFront()
    
    # records statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    # execute the basic evolutionry loop with given components
    algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=1000, stats=stats, halloffame=hof)
    return pop, stats, hof
                 
if __name__ == "__main__":
    pop, stats, hof = main()                 
    # print the fitness of the best solution in the hof
    print(hof[0].fitness.values)