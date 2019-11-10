import numpy as np
import trex_nn

import time
import logging
from DinoGameSession import DinoGameSession
from Config import Config
###### PART 1: PREPARE FIXED VARs AND FUNCTIONs ######
######################################################
######################################################

def cv_to_sequence(body):
	"""
	cv body aka parameters_set to ADN aka sequence
	"""
	sequence = []
	for key in Config.BODY_KEYS:
		sequence.append(body[key])
	sequence_str = str(sequence)
	sequence_str = sequence_str.replace("[", "").replace("array", "")
	sequence_str = sequence_str.replace("]", "").replace("\n", "")
	sequence_str = sequence_str.replace("(", "").replace(")", "")
	sequence_str = sequence_str.replace(" ", "")
	sequence_adj = list(map(float, sequence_str.split(",")))
	return sequence_adj


def cv_to_body(adn):
	"""
	cv ADN aka sequence to body aka parameters_set
	tam thoi hard code 1 ty chu ko chac chet luon =))
	"""
	params = {}
	params["W1"] = np.reshape(adn[:9], (3, 3))
	params["W2"] = np.reshape(adn[9:12], (1, 3))
	params["b1"] = np.reshape(adn[12:15], (3, 1))
	params["b2"] = np.reshape(adn[15], (1, 1))
	return params


def genesis(pop_size = Config.POP_SIZE):
	trex_clan = [trex_nn.initialize_parameters(Config.N_X, Config.N_H, Config.N_Y) for i in range(Config.POP_SIZE)]
	trex_clan = np.array(trex_clan)
	return trex_clan


def random_match(random_set = Config.RANDOM_SET):
	number = -1
	curr_number = sum(Config.RANDOM_SET)
	lucky_number = np.random.randint(0, curr_number)
	while lucky_number < curr_number:
		number += 1
		curr_number -= Config.RANDOM_SET[number]
	return number


def do_mutation(child, mutation_prob = Config.MUTATION_PROB):
	mutation_rate = np.random.random(16)
	new_child = child[:]
	for ind in range(16):
		if mutation_rate[ind] < mutation_prob:
			if ind < 9:
				new_child[ind] += np.random.randn() * Config.MUTATION_RANGE[0]
			elif ind < 12:
				new_child[ind] += np.random.randn() * Config.MUTATION_RANGE[1]
			elif ind < 15:
				new_child[ind] += np.random.randn() * Config.MUTATION_RANGE[2]
			else:
				new_child[ind] += np.random.randn() * Config.MUTATION_RANGE[3]
	return new_child


def crossver(adam, eva):
	weight_adam = adam[:12]
	weight_eva = eva[:12]
	bias_adam = adam[12:]
	bias_eva = eva[12:]

	cut_1 = np.random.randint(0, 12)
	cut_2 = np.random.randint(12, 16) - 12

	childs = []
	childs.append(weight_adam[:cut_1] + weight_eva[cut_1:] + bias_adam[:cut_2] + bias_eva[cut_2:])
	childs.append(weight_adam[:cut_1] + weight_eva[cut_1:] + bias_eva[:cut_2] + bias_adam[cut_2:])
	childs.append(weight_eva[:cut_1] + weight_adam[cut_1:] + bias_adam[:cut_2] + bias_eva[cut_2:])
	childs.append(weight_eva[:cut_1] + weight_adam[cut_1:] + bias_eva[:cut_2] + bias_adam[cut_2:])
	selected_child = np.random.randint(0, 4)
	return childs[selected_child]


def breed_a_child(survivals):
	"""
	chon 2 cha me tu survivals va crossver
	sau do cho child mutation
	"""
	adam_ind = random_match(Config.RANDOM_SET)
	eva_ind = random_match(Config.RANDOM_SET)
	# boi vi so ok kha it --> co the cu de tu breed cung dc di
	# while adam_ind == eva_ind:
	#	 eva_ind = random_match(RANDOM_SET)
	expected_cain = crossver(survivals[adam_ind], survivals[eva_ind])
	real_cain = do_mutation(expected_cain)
	return real_cain


def gen_to_max_size(survivals, pop_size = Config.POP_SIZE):
	"""

	"""
	curr_len = Config.N_SIZE
	dna_survivals = list(map(cv_to_sequence, survivals))
	dna_tribal = dna_survivals[:]
	while curr_len < pop_size:
		new_born = breed_a_child(dna_survivals)
		dna_tribal.append(new_born)
		curr_len += 1
	new_gen = list(map(cv_to_body, dna_tribal))
	new_gen = np.array(new_gen)
	return new_gen


def select_survivals(tribal, score):
	fitness_scores = np.array(score)
	survival_inds = (-fitness_scores).argsort()[:4]
	return tribal[survival_inds], survival_inds



######  PART 2: EVOLVE						  ######
######################################################
######################################################

def evolve():
	adam_eva = genesis(Config.POP_SIZE)
	curr_gen = adam_eva.copy()
	count_gen = 0
	finest = 0
	while True:
		log1.info("-------------------------------------------------------")
		log2.info("-------------------------------------------------------")
		log1.info("Generation: {}".format(count_gen))
		score = []
		for i, trex in enumerate(curr_gen):
			game = DinoGameSession()
			game.play(trex,f'Generation: {count_gen}. Population: {i+1}/{Config.POP_SIZE}. Finest: {finest}')
			score.append(game.score)
			finest = max(max(score),finest)
		log1.info(score)
		survivals, survival_inds = select_survivals(curr_gen, score)
		log1.info(survival_inds)
		log1.info("Genarating next gen")
		curr_gen = gen_to_max_size(survivals, Config.POP_SIZE)
		count_gen += 1
		log2.info(survivals)

		# if max(score) > 15:
		# 	log1.info(survivals)
		# 	break
		# elif count_gen % 10 == 0:
		# 	log1.info(survivals)
		# log1.info("")


if __name__ == '__main__':
	Config.init()
	log1 = Config.log_brief
	log2 = Config.log_survival

	evolve()