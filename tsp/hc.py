import evaluate
import random
from tqdm import tqdm

def get_conseq_neighbours(tour):
	neighbours = []
	for i in range(126):
		n = list(tour)
		n[i], n[i+1] = n[i+1], n[i]
		neighbours.append(n)
	return neighbours

def get_neighbours(tour):
	# we will generate 100 random neighbours
	neighbours = []
	for i in range(100):
		n = list(tour)
		locs = random.choices(tour, k=2)
		n[locs[0]], n[locs[1]] = n[locs[1]], n[locs[0]]
		neighbours.append(n)
	return neighbours

# random starting point
current = list(range(127))
random.shuffle(current)
current_fitness = evaluate.evaluate(current)

climb = True
while climb:
	climb = False
	# get neighbours
	neighbours = get_neighbours(current)
	# sort according to fitness to do steepest ascent
	neighbours = sorted(neighbours, key=lambda x: evaluate.evaluate(x))
	best_neighbour_fitness = evaluate.evaluate(neighbours[0])
	# climb if possible
	if best_neighbour_fitness < current_fitness:
		current = list(neighbours[0])
		current_fitness = best_neighbour_fitness
		print(current_fitness)
		climb = True
print(current_fitness)
