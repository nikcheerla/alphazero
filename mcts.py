
import sys, os, random, time, warnings

from itertools import product
import numpy as np
import IPython

from scipy.stats import pearsonr
from utils import hashable, sample

import multiprocessing
from multiprocessing import Pool, Manager






class MCTSController(object):

	def __init__(self, manager, T=0.3, C=1.5):
		super().__init__()

		self.visits = manager.dict()
		self.differential = manager.dict()
		self.T = T
		self.C = C

	def record(self, game, score):
		self.visits["total"] = self.visits.get("total", 1) + 1
		self.visits[hashable(game.state())] = self.visits.get(hashable(game.state()), 0) + 1
		self.differential[hashable(game.state())] = self.differential.get(hashable(game.state()), 0) + score

	r"""
	Runs a single, random heuristic guided playout starting from a given state. This updates the 'visits' and 'differential'
	counts for that state, as well as likely updating many children states.
	"""
	def playout(self, game, expand=150):

		if expand == 0 or game.over():
			score = game.score()
			self.record(game, score)
			#print ('X' if game.turn==1 else 'O', score)
			return score

		action_mapping = {}

		for action in game.valid_moves():
			
			game.make_move(action)
			action_mapping[action] = self.heuristic_value(game)
			game.undo_move()

		chosen_action = sample(action_mapping, T=self.T)
		game.make_move(chosen_action)
		score = -self.playout(game, expand=expand-1) #play branch
		game.undo_move()
		self.record(game, score)

		return score

	r"""
	Evaluates the "value" of a state as a bandit problem, using the value + exploration heuristic.
	"""
	def heuristic_value(self, game):
		N = self.visits.get("total", 1)
		Ni = self.visits.get(hashable(game.state()), 1e-9)
		V = self.differential.get(hashable(game.state()), 0)*1.0/Ni 
		return V + self.C*(np.log(N)/Ni)

	r"""
	Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
	"""
	def value(self, game, playouts=100, steps=5):

		# play random playouts starting from that game value
		with Pool() as p:
			scores = p.map(self.playout, [game.copy() for i in range(0, playouts)])

		return self.differential[hashable(game.state())]*1.0/self.visits[hashable(game.state())]

	r"""
	Chooses the move that results in the highest value state.
	"""
	def best_move(self, game, playouts=100):

		action_mapping = {}

		for action in game.valid_moves():
			game.make_move(action)
			action_mapping[action] = self.value(game, playouts=playouts)
			game.undo_move()

		print ({a: "{0:.2f}".format(action_mapping[a]) for a in action_mapping})
		return max(action_mapping, key=action_mapping.get)







