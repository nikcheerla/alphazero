
import sys, os, random, time

from itertools import product
import numpy as np
import chess

import IPython

from multiprocessing import Pool
from representations import state, board
from utils import hashable





visits = {}
differential = {}

def record(game, score):
	visits["total"] = visits.get("total", 1) + 1
	visits[state(game)] = visits.get(state(game), 0) + 1
	differential[state(game)] = differential.get(state(game), 0) + score
	#print (game)
	#print ()
	#print (differential[state(game)], visits[state(game)])
	#print ()



r"""
Runs a single, random heuristic guided playout starting from a given state. This updates the 'visits' and 'differential'
counts for that state, as well as likely updating many children states.
"""

def playout(game=chess.Board(), C=1.5, expand=150):

	if expand == 0 or game.is_game_over():
		score = 1 if game.is_checkmate() else 0
		record(game, score)
		return score

	action_mapping = {}

	for action in game.generate_legal_moves():
		game.push(action)

		N = visits.get("total", 1)
		Ni = visits.get(state(game), 1e-5)
		V = differential.get(state(game), 0)*1.0/Ni
		U = V + C*(np.log(N)/Ni)
		U = min(U, 100000)
		action_mapping[action] = U + random.random()*1e-5 #tiebreaker

		game.pop()

	chosen_action = max(action_mapping, key=action_mapping.get)
	
	game.push(chosen_action)
	score = -playout(game, C=C, expand=expand-1) #play branch
	game.pop()
	record(game, score)

	return score



r"""
Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
"""

def value(game=chess.Board(), playouts=100):

	# play random playouts starting from that game value
	for i in range(0, playouts):
		playout(game=game)
	return differential[state(game)]*1.0/visits[state(game)]



r"""
Chooses the move that results in the highest value state.
"""

def best_move(game=chess.Board(), playouts=100):

	action_mapping = {}
	move_pool = Pool()
	for action in game.generate_legal_moves():
		game.push(action)
		action_mapping[action] = move_pool.apply_async(value, (game, playouts))
		game.pop()

	return max(action_mapping, key=lambda x: action_mapping[x].get())








if __name__ == "__main__":

	game = chess.Board()
	print (game)
	print ()

	for i in range(1, 100000):
		game.push(best_move(game, playouts=10))

		print (game)
		print ()

