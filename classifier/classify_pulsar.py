#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

training_xs = []
training_ys = []

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedLog(x):
    return math.log(abs(x))

ARGS_SIZE = 8
pset = gp.PrimitiveSet("MAIN", ARGS_SIZE)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
# pset.renameArguments(ARG0='x')
args_map = dict(
    [("ARG" + str(i), "x" + str(i + 1)) for i in range(ARGS_SIZE)])
pset.renameArguments(**args_map)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalClassifier(individual):
    global training_xs, training_ys
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    fitness = 0

    return fitness,

toolbox.register("evaluate", evalClassifier)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def read_csv(file_path, balance=False):
    xs = []
    ys = []

    pxs = []
    nxs = []

    with open(file_path, "r") as f:
        for line in f.readlines():
            temp = line.strip().split(",")
            x = temp[:-1]
            y = temp[-1]
            for i in range(len(x)):
                x[i] = float(x[i])
            if int(y) == 1:
                pxs.append(x)
            else:
                nxs.append(x)
    if balance:
        xs.extend(pxs)
        ys.extend([1] * len(pxs))

        xs.extend(nxs[:len(pxs)])
        ys.extend([0] * len(pxs))

        assert len(xs)==len(ys)
        print("Data Size:", len(xs), "rows")
        return xs, ys
    
    else:
        xs.extend(pxs)
        ys.extend([1] * len(pxs))

        xs.extend(nxs)
        ys.extend([0] * len(nxs))

        assert len(xs)==len(ys)
        print("Data Size:", len(xs), "rows")
        return xs, ys



def main():
    # random.seed(318)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    print(hof[0])
    return pop, log, hof

def split_data():
    n = 17898
    half = n / 2

    train_data = []
    test_data = []

    with open("./pulsar_stars.csv", "r") as f:
        i = 0
        for line in f.readlines()[1:]:
            # temp = line.strip().split(",")
            if i > half:
                test_data.append(line)
            else:
                train_data.append(line)
            i += 1
    
    with open("./pulsar_train.csv", "w") as f:
        for row in train_data:
            f.write(row)
    
    with open("./pulsar_test.csv", "w") as f:
        for row in test_data:
            f.write(row)


if __name__ == "__main__":
    training_xs, training_ys = read_csv("./pulsar_train.csv", balance=True)
    pop, log, hof = main()
    
    best = hof[0]

    predictor = toolbox.compile(expr=best)

    test_xs, test_ys = read_csv("./pulsar_test.csv")
    predicted_ys = []
    for row in test_xs:
        pred = 1 if predictor(*row) > 0 else 0
        predicted_ys.append(pred)
    
    print(test_ys)
    print(predicted_ys)

    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    
    precision = precision_score(test_ys, predicted_ys)
    recall = recall_score(test_ys, predicted_ys)

    print('Precision score: {0:0.2f}'.format(precision))
    print('Recall score: {0:0.2f}'.format(recall))

    # split_data()

    