
import sys, os, random, time

from itertools import product
import numpy as np
from tictactoe import *

import IPython

from multiprocessing import Pool
from utils import hashable, sample



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
		V = differential.get(tuple(game), 0)*1.0/Ni
		U = V + C*(np.log(N)/Ni)
		U = min(U, 100000)
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

def value(game=[' ']*10, letter='X', playouts=100):

	# play random playouts starting from that game value
	for i in range(0, playouts):
		playout(game=game, letter=letter)
	return differential[tuple(game)]*1.0/visits[tuple(game)]



r"""
Chooses the move that results in the highest value state.
"""

def best_move(game=[' ']*10, letter='X', playouts=100):

	action_mapping = {}
	#move_pool = Pool()
	for action in validMoves(game):
		makeMove(game, letter, action)
		action_mapping[action] = value(game, turn(letter), playouts) #move_pool.apply_async(value, (game, turn(letter), playouts))
		undoMove(game)

	#print ({k: action_mapping[k].get() for k in action_mapping.keys()})
	return max(action_mapping, key=lambda x: action_mapping[x])#.get())









if __name__ == "__main__":

	for i in range(0, 10):

		game = [' ']*10
		letter = 'X'
		drawBoard(game)
		print ()

		while not isGameOver(game):
			makeMove(game, letter, best_move(game, letter=turn(letter), playouts=100))
			drawBoard(game)

			letter = turn(letter)
			print ()
			time.sleep(0.5)

		print ()
		print ()
		print ()











