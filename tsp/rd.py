import evaluate
import random
from tqdm import tqdm

best_fitness = 1000000
for i in tqdm(range(1000000)):
	tour = list(range(127))
	random.shuffle(tour)
	fitness = evaluate.evaluate(tour)
	if fitness < best_fitness:
		best_tour = list(tour)
		best_fitness = fitness
		# print(best_fitness)
print(best_tour)
print(best_fitness)