
import sys, os, random, time

from itertools import product
import numpy as np
from tictactoe import *

import IPython

from multiprocessing import Pool
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from utils import hashable, sample

from modules import TrainableModel

import torch
import torch.nn as nn
import torch.nn.functional as F


""" Network
"""

class Net(TrainableModel):

	def __init__(self):

		super(TrainableModel, self).__init__()
		self.layer1 = nn.Linear(10, 128)
		self.layer2 = nn.Linear(128, 128)
		self.layer3 = nn.Linear(128, 1)
		

	def loss(self, data, data_pred):
		Y_pred = data_pred["target"]
		Y_target = data["target"]

		return F.mse_loss(Y_pred, Y_target)

	# Subclasses: please override depending on what type of scoring is required
	def score(self, preds, targets):
		return
		#preds['target'][0, 0] += random.random()*1e-6
		score, _ = pearsonr(preds['target'][:, 0], targets['target'])
		base_score, _ = pearsonr(preds['target'][:, 0], shuffle(targets['target']))
		return "{0:.4f}/{1:.4f}".format(score, base_score)       

	def forward(self, x):
		x = self.layer1(x['input'])
		x = self.layer2(F.tanh(x))
		x = F.dropout(x, p=0.1, training=self.training)
		x = self.layer3(F.relu(x))
		return {'target': x}

model = Net()
model.compile(torch.optim.Adadelta, lr=0.3)


visits = {}
differential = {}

def record(game, score):
	visits["total"] = visits.get("total", 1) + 1
	visits[tuple(game)] = visits.get(tuple(game), 0) + 1
	differential[tuple(game)] = differential.get(tuple(game), 0) + score



r"""
Runs a single, random heuristic guided playout starting from a given state. This updates the 'visits' and 'differential'
counts for that state, as well as likely updating many children states.
"""

def playout(game=[' ']*10, letter='X', C=1.5):
	
	if isGameOver(game):
		score = -getScore(game, letter)
		record(game, score)
		return score

	action_mapping = {}

	for action in validMoves(game):
		makeMove(game, letter, action)

		N = visits.get("total", 1)
		Ni = visits.get(tuple(game), 1e-5)

		game_data = (np.array(game)=='X').astype(float) - (np.array(game)=='O').astype(int)
		V = model.predict([{'input': game_data, 'target': None}]).mean()
		#V = differential.get(tuple(game), 0)*1.0/Ni
		U = V + C*(np.log(N)/Ni)
		U = min(U, 100000)
		U += 1
		action_mapping[action] = (U + random.random())**2 #tiebreaker

		undoMove(game)
	
	chosen_action = sample(action_mapping)
	makeMove(game, letter, chosen_action)
	score = -playout(game, letter=turn(letter))
	undoMove(game)
	record(game, score)

	return score



r"""
Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
"""

def value(game=[' ']*10, letter='X', playouts=100, pred=False):

	# play random playouts starting from that game value
	for i in range(0, playouts):
		playout(game=game, letter=letter)
	V = differential[tuple(game)]*1.0/visits[tuple(game)]

	game_data = (np.array(game)=='X').astype(float) - (np.array(game)=='O').astype(float)
	dataset = [{'input': game_data, 'target': V}]
	for i in range(0, 5):
		model.fit(dataset, batch_size=1, verbose=False)
	if pred: V = model.predict(dataset).mean()

	return V



r"""
Chooses the move that results in the highest value state.
"""

def best_move(game=[' ']*10, letter='X', playouts=100):

	action_mapping = {}
	network_mapping = {}
	#move_pool = Pool()
	for action in validMoves(game):
		makeMove(game, letter, action)
		action_mapping[action] = value(game, turn(letter), playouts) #move_pool.apply_async(value, (game, turn(letter), playouts))
		network_mapping[action] = value(game, turn(letter), playouts, pred=True)
		undoMove(game)

	#print ({k: action_mapping[k].get() for k in action_mapping.keys()})
	print ({a: "{0:.4f}".format(action_mapping[a]) for a in action_mapping})
	print ({a: "{0:.4f}".format(network_mapping[a]) for a in action_mapping})

	moves = validMoves(game)
	data1 = [action_mapping[action] for action in moves]
	data2 = [network_mapping[action] for action in moves]
	R, p = pearsonr(data1, data2)
	if len(data1) > 2: print ("Correlation: {0:.4f} (p={1:.4f})".format(R, p))

	return max(action_mapping, key=lambda x: action_mapping[x])









if __name__ == "__main__":

	for i in range(0, 100):

		game = [' ']*10
		letter = 'X'
		drawBoard(game)
		print ()

		while not isGameOver(game):
			makeMove(game, letter, best_move(game, letter=letter, playouts=20))
			drawBoard(game)

			letter = turn(letter)
			print ()
			time.sleep(0.5)

		print ()
		print ()
		print ()











