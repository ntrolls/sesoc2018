# KIISE SE Society 2018 Summer Lecture on SBSE

This repository contains slides and other course materials for the [short summer lecture on Search Based Software Engineering (SBSE)](http://www.kiise.or.kr/conference/conf/022/), hosted by the Software Engineering Society of KIISE.

## Dependencies

The hands-on materials have the following dependencies:

- Java 8 Runtime
- Python (provided materials are written for Python 3 but version dependency is not significant)
- [DEAP](https://github.com/DEAP/deap): Distributed Evolutionary Algorithms in Python. You can install via `pip`: `pip install deap`
- [numpy](http://www.numpy.org): `pip install numpy`
- [scipy](https://www.scipy.org): `pip install scipy`

## Hands-on 1: Travelling Salesman Problem Hands-on

The goal is to solve a TSP instance. An utility module, [`evaluate.py`](tsp/evaluate.py), and a data file [`bier127.dat`](tsp/bier127.dat) that contains a TSP instance (you can find the original file at [`bier127.tsp`](http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/bier127.tsp)), are available from this repository. It contains coordinates of 127 beer gardens in [Ausburg](https://en.wikipedia.org/wiki/Augsburg). Your goal is to find the shortest route that passes all the beer gardens!

The beer gardens have indices from 0 to 126. The shortest route is one permutation out of 127! possibilities. If you submit a permutation, i.e., a Python `list` of integers in [0, 126], to the function `evaluate` in `evaluate.py`, you will get the length of the route. For example, the following code snippet will give you the length of a random route:

```python
import evaluate from evaluate
import random

random_route = list(range(127))
random.shuffle(random_route)
print(evaluate(random_route))
```

Try finding the shortest route using any method (but, preferably, using a metaheuristic based on what we have covered so far). We will walk through a GA based solver during the later part of the hands-on session.

### Reference
- Refer to [DEAP](https://github.com/DEAP/deap) and its [documentations](https://deap.readthedocs.io/en/master/) for further details.

## Hands-on 2: Evolving Spectrum Based Fault Localisation Formulas

The goal is to evolve a risk evaluation formula, which forms the core of Spectrum Based Fault Localisation. This is a small scale replication of the paper [Evoling Human Competitive Spectra-Based Fault Localisation Techniques](https://link.springer.com/chapter/10.1007/978-3-642-33119-0_18) (the [PDF](sbfl/paper.pdf) is in this repository).

It is recommended that participants take a look at the symbolic regression Genetic Programming code, written using [DEAP](https://github.com/DEAP/deap). Focus on the following parts:

### Setting up GP node types

```python
pset = gp.PrimitiveSet("MAIN", 1) # GP trees use 1 input variable
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
```

We use `PremitiveSet` to define available GP node types for individuals.

### Fitness Evaluation for Symbolic Regression

```python
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),
```

The target equation in this example is known: it is hard-coded in the fitness evaluation function: `x**4 + x**3 + x**2 + x`. The above function computes the Mean Squared Error between the given GP tree and the target function, using `x` values in `points`.

### Modifying `symreg.py` for SBFL

There are a number of important differences:

- We use GP to evolve formulas that use four input variables: `ep, ef, np nf`.
- Fitness should be computed using ranks of known faults.
- We do not have an absolute evaluation metric such as MSE: instead, the quality of learning should be evaluated using unseen faults.

#### Setting up `PrimitiveSet`

```python
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
```

Note that we use four variables now, as well as a different set of GP operators (mostly because we do not think `sine` and `cosine` will make sense for localisation).

#### Filling in fitness evaluation

```python
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
```

The main focus of your hands-on should be constructing the fitness function. The comments will guide you through the process. Once you complete the function and can run the GP, try changing various parameters or the fitness itself.
