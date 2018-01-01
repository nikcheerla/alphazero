
import sys, os, random

from itertools import product
import numpy as np
import chess

from games.games import AbstractGame

import IPython


class GomokuGame(AbstractGame):

	def __init__(self):
		super().__init__()
		self.W = 6
		self.board = np.zeros((self.W, self.W))
		self.turn = 1 # Implies player 1 is to play first move
		self.move_stack = []

	def valid_moves(self):
		return [(i, j) for i, j in product(range(0, self.W), range(0, self.W)) if self.board[i, j] == 0]

	def over(self):
		if self.score() != 0: return True
		return np.all(self.board != 0)

	def is_winner(self, player):
		
		for i, j in product(range(0, self.W), range(0, self.W)):
			for dx, dy in product([-1, 0, 1], [-1, 0, 1]):
				if dx == 0 and dy == 0: continue

				five = True
				for M in [-2, -1, 0, 1, 2]:
					if i+M*dx < 0 or i+M*dx >= self.W or j+M*dy < 0 or j+M*dy >= self.W:
						five = False
						continue
					if self.board[i+M*dx, j+M*dy] != player:
						five = False
				if five: return True
		return False

	def score(self):
		if self.is_winner(1):
			return -1 if self.turn == 1 else 1
		if self.is_winner(-1):
			return -1 if self.turn == -1 else 1
		return 0

	def make_move(self, move):
		self.board[move] = self.turn
		self.move_stack.append(move)
		self.turn *= -1

	def undo_move(self):
		self.board[self.move_stack.pop()] = 0
		self.turn *= -1

	def state(self):
		data = np.zeros((2, self.W, self.W))
		data[0, :, :] = self.turn
		data[1, :, :] = self.board.copy()
		return data

	@classmethod
	def load(cls, state):

		game = GomokuGame()
		game.turn = round(state.copy()[0].mean())
		game.board = state.copy()[1]
		return game

	def __str__(self):
		return '\n'.join([''.join(['X' if num==1 else 'O' if num==-1 else '-' for num in nums]) for nums in self.board])

	def __repr__(self):
		return repr(self.board)





if __name__ == "__main__":
	
	game = GomokuGame()
	game = GomokuGame.load(game.state())
	print (game)
	#print (repr(game))
	print ()

	while not game.over():
		actions = list(game.valid_moves())
		game.make_move(random.choice(actions))

		game = GomokuGame.load(game.state())

		print (game)
		#print (repr(game))
		print ()

	print (game.score())
	print (game.turn)


