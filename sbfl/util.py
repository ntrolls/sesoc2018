import os
import scipy.stats as ss

def read_training_data():
	training_data = []
	for _filename in os.listdir("./data/training"):
		with open(os.path.join("./data/training", _filename), "r") as f:
			lines = f.readlines()
			faulty_index = int(lines[0].strip())
			spectrum = []
			for row in lines[1:]:
				ep, ef, np, nf = [int(x) for x in row.strip().split(",")]
				spectrum.append((ep, ef, np, nf))
			training_data.append((faulty_index, spectrum))
	return training_data

def read_test_data():
	test_data = []
	for _filename in os.listdir("./data/test"):
		with open(os.path.join("./data/test", _filename), "r") as f:
			lines = f.readlines()
			faulty_index = int(lines[0].strip())
			spectrum = []
			for row in lines[1:]:
				ep, ef, np, nf = [int(x) for x in row.strip().split(",")]
				spectrum.append((ep, ef, np, nf))
			test_data.append((faulty_index, spectrum))
	return test_data

def ranking(faulty_index, scores):
	return ss.rankdata(scores, method="max")[faulty_index]