
import sys, os, random

from itertools import product
import numpy as np
import chess

from games.games import AbstractGame

import IPython




class TicTacToeGame(AbstractGame):

	def __init__(self):
		super().__init__()
		self.board = [' ']*10
		self.turn = 'X' #implies that X is to play next move
		self.move_stack = []

	def valid_moves(self):
		return [i for i in range(1, 10) if self.board[i] == ' ']

	def over(self):
		if self.score() != 0: return True
		return all(b != ' ' for b in self.board[1:10])

	def is_winner(self, letter):
		# Given a board and a player's letter, this function returns True if that player has won.
		# We use bo instead of board and le instead of letter so we don't have to type as much.
		return ((self.board[7] == letter and self.board[8] == letter and self.board[9] == letter) or # across the top
		(self.board[4] == letter and self.board[5] == letter and self.board[6] == letter) or # across the middle
		(self.board[1] == letter and self.board[2] == letter and self.board[3] == letter) or # across the bottom
		(self.board[7] == letter and self.board[4] == letter and self.board[1] == letter) or # down the left side
		(self.board[8] == letter and self.board[5] == letter and self.board[2] == letter) or # down the middle
		(self.board[9] == letter and self.board[6] == letter and self.board[3] == letter) or # down the right side
		(self.board[7] == letter and self.board[5] == letter and self.board[3] == letter) or # diagonal
		(self.board[9] == letter and self.board[5] == letter and self.board[1] == letter)) # diagonal

	def score(self):
		if self.is_winner('X'):
			return -1 if self.turn == 'X' else 1
		if self.is_winner('O'):
			return -1 if self.turn == 'O' else 1
		return 0

	def make_move(self, move):
		self.board[move] = self.turn
		self.move_stack.append(move)
		self.turn = 'X' if self.turn == 'O' else 'O'

	def undo_move(self):
		self.board[self.move_stack.pop()] = ' '
		self.turn = 'X' if self.turn == 'O' else 'O'

	def state(self):
		self.board[0] = self.turn
		data = (np.array(self.board)=='X').astype(float) - (np.array(self.board)=='O').astype(float)
		data2 = np.zeros((2, 3, 3))
		data2[0, :, :] = data[0]
		data2[1, :, :] = data[1:10].reshape(3, 3)
		return data2

	@classmethod
	def load(cls, state):

		data = np.zeros(10)
		data[0] = round(state[0].mean())
		data[1:10] = state[1].flatten()
		state = data

		game = TicTacToeGame()

		game.board = np.array(game.board)
		game.board[state==1] = 'X'
		game.board[state==-1] = 'O'
		game.board = game.board.tolist()

		game.turn = game.board[0]
		game.board[0] = ' '
		return game

	def __str__(self):

		string = ""
		string += ('   |   |') + "\n"
		string += (' ' + self.board[7] + ' | ' + self.board[8] + ' | ' + self.board[9]) + "\n"
		string += ('   |   |') + "\n"
		string += ('-----------') + "\n"
		string += ('   |   |') + "\n"
		string += (' ' + self.board[4] + ' | ' + self.board[5] + ' | ' + self.board[6]) + "\n"
		string += ('   |   |') + "\n"
		string += ('-----------') + "\n"
		string += ('   |   |') + "\n"
		string += (' ' + self.board[1] + ' | ' + self.board[2] + ' | ' + self.board[3]) + "\n"
		string += ('   |   |') + "\n"
		return string

	def __repr__(self):
		return repr(self.board)





if __name__ == "__main__":
	
	game = TicTacToeGame()
	game = TicTacToeGame.load(game.state())
	print (game)
	print (repr(game))
	print ()

	while not game.over():
		actions = list(game.valid_moves())
		game.make_move(random.choice(actions))

		game = TicTacToeGame.load(game.state())

		print (game)
		print (repr(game))
		print ()


