
import sys, os, random, time

from itertools import product
import numpy as np
import chess

import IPython

from representations import state, board
from scipy.stats import pearsonr
from utils import hashable, sample

import multiprocessing
from multiprocessing import Pool, Manager

from modules import TrainableModel

import torch
import torch.nn as nn
import torch.nn.functional as F


""" CNN representing estimated value for each board state.
"""

class ChessNet(TrainableModel):

	def __init__(self):

		super(TrainableModel, self).__init__()
		self.conv1 = nn.Conv2d(11, 32, kernel_size=(3, 3), padding=(1, 1))
		self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
		self.conv3= nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), dilation=2)
		self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(2, 2), dilation=2)
		self.linear = nn.Linear(64, 1)
	
	def loss(self, data, data_pred):
		Y_pred = data_pred["target"]
		Y_target = data["target"]

		return F.mse_loss(Y_pred, Y_target)

	# Subclasses: please override depending on what type of scoring is required
	def score(self, preds, targets):
		#preds['target'][0, 0] += random.random()*1e-6
		score, _ = pearsonr(preds['target'][:, 0], targets['target'])
		base_score, _ = pearsonr(preds['target'][:, 0], shuffle(targets['target']))
		return "{0:.4f}/{1:.4f}".format(score, base_score)

	def forward(self, x):
		x = x['input']
		x = x.permute(0, 3, 1, 2)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.dropout(x, p=0.2, training=self.training)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = x.mean(dim=2).mean(dim=2)
		x = self.linear(x)
		return {'target': x}




class MCTSController(object):

	def __init__(self, manager, model):
		self.visits = manager.dict()
		self.differential = manager.dict()
		self.model = model
		super(MCTSController, self).__init__()

	def record(self, game, score):
		self.visits["total"] = self.visits.get("total", 1) + 1
		self.visits[hashable(state(game))] = self.visits.get(hashable(state(game)), 0) + 1
		self.differential[hashable(state(game))] = self.differential.get(hashable(state(game)), 0) + score

	r"""
	Runs a single, random heuristic guided playout starting from a given state. This updates the 'visits' and 'differential'
	counts for that state, as well as likely updating many children states.
	"""
	def playout(self, game=chess.Board(), C=1.5, expand=80):

		if expand == 0 or game.is_game_over():
			score = 1 if game.is_checkmate() else 0
			self.record(game, score)
			print (score, file=sys.stderr)
			return score

		action_mapping = {}

		for action in game.generate_legal_moves():
			game.push(action)

			N = self.visits.get("total", 1)
			Ni = self.visits.get(hashable(state(game)), 1e-5)
			#V = self.differential.get(hashable(state(game)), 0)*1.0/Ni
			V = self.network_value(game)
			U = V + C*(np.log(N)/Ni)
			U = min(U, 100000) + 1
			action_mapping[action] = (U + random.random()*1e-5)**2 #tiebreaker

			game.pop()

		chosen_action = sample(action_mapping)
		game.push(chosen_action)
		score = -self.playout(game, C=C, expand=expand-1) #play branch
		game.pop()
		self.record(game, score)

		return score

	r"""
	Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
	"""
	def value(self, game=chess.Board(), playouts=100, steps=5):

		# play random playouts starting from that game value
		with Pool() as p:
			p.map(self.playout, [board(state(game)) for i in range(0, playouts)])

		V = self.differential[hashable(state(game))]*1.0/self.visits[hashable(state(game))]
		dataset = [{'input': state(game), 'target': V}]
		for i in range(0, steps):
			self.model.fit(dataset, batch_size=1, verbose=False)

		return V

	r"""
	Evaluates the "value" of a state using the network estimation function.
	"""
	def network_value(self, game=chess.Board()):
		dataset = [{'input': state(game), 'target': None}]
		return self.model.predict(dataset).mean()

	r"""
	Chooses the move that results in the highest value state.
	"""
	def best_move(self, game=chess.Board(), playouts=100):

		action_mapping = {}
		previous_mapping = {}
		network_mapping = {}

		for action in game.generate_legal_moves():
			game.push(action)
			previous_mapping[action] = self.network_value(game)
			action_mapping[action] = self.value(game, playouts=playouts)
			network_mapping[action] = self.network_value(game)
			game.pop()

		print ({a.uci(): "{0:.4f}".format(action_mapping[a]) for a in action_mapping})

		moves = action_mapping.keys()
		data1 = [action_mapping[action] for action in moves]
		data2 = [previous_mapping[action] for action in moves]
		data3 = [network_mapping[action] for action in moves]
		R1, p1 = pearsonr(data1, data2)
		R2, p2 = pearsonr(data1, data3)
		print ("Correlation before fitting: {0:.4f} (p={1:.4f})".format(R1, p1))
		print ("Correlation after fitting: {0:.4f} (p={1:.4f})".format(R2, p2))

		return max(action_mapping, key=action_mapping.get)







if __name__ == "__main__":

	
	manager = Manager()
	model = ChessNet()
	model.compile(torch.optim.Adadelta, lr=0.3)
	controller = MCTSController(manager, model)

	
	for i in range(0, 1000):
		game = chess.Board()
		print (game)
		print ()

		while not game.is_game_over():
			game.push(controller.best_move(game, playouts=2))

			print (game)
			print ()

