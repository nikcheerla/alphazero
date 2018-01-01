
import sys, os, random

from itertools import product
import numpy as np
import chess

from games.games import AbstractGame

import IPython


pieces = ["P", "N", "B", "R", "Q", "K"]



class ChessGame(AbstractGame):

	def __init__(self, fen=chess.Board().fen()):
		super().__init__()
		self.game = chess.Board(fen)

	def valid_moves(self):
		return self.game.generate_legal_moves()

	def over(self):
		return self.game.is_game_over()

	def score(self):
		return 1 if self.game.is_checkmate() else 0

	def make_move(self, move):
		self.game.push(move)

	def undo_move(self):
		self.game.pop()

	def state(self):
		
		positions = [line.split(" ") for line in str(self.game).split("\n")]
		H, W = len(positions), len(positions[0])
		state = np.zeros((H, W, len(pieces)+5))
		for i, j in product(range(H), range(W)):
			if positions[i][j].upper() in pieces:
				# White pieces are +1, black pieces are -1
				state[i, j, pieces.index(positions[i][j].upper())] = +1 if positions[i][j].isupper() else -1
		turn = 1 if self.game.turn else 0   #1 if white's turn, 0 if white's turn
		KW, KB = self.game.has_kingside_castling_rights(chess.WHITE), self.game.has_kingside_castling_rights(chess.BLACK)
		QW, QB = self.game.has_queenside_castling_rights(chess.WHITE), self.game.has_queenside_castling_rights(chess.BLACK)
		state[:, :, len(pieces)].fill(turn)
		state[:, :, len(pieces)+1].fill(1 if KW else 0)
		state[:, :, len(pieces)+2].fill(1 if QW else 0)
		state[:, :, len(pieces)+3].fill(1 if KB else 0)
		state[:, :, len(pieces)+4].fill(1 if QB else 0)

		return state

	@classmethod
	def load(cls, state):
		
		data = [['.']*8 for i in range(0, 8)]
		(H, W, P) = state.shape
		for i, j, k in product(range(H), range(W), range(P-5)):
			if state[i, j, k] != 0:
				data[i][j] = pieces[k] if state[i, j, k] == 1 else pieces[k].lower()

		data = [''.join(line) for line in data]
		game = []

		for line in data:
			count = 0
			line2 = ""
			for c in line:
				if c == '.':
					count += 1
				else:
					if (count != 0): line2 += str(count)
					line2 += c
					count = 0
			if (count != 0): line2 += str(count)
			game.append(line2)

		turn = 'w' if round(state[:, :, P-5].mean()) == 1 else 'b'
		KW = 'K' if round(state[:, :, P-4].mean()) == 1 else ''
		QW = 'Q' if round(state[:, :, P-3].mean()) == 1 else ''
		KB = 'k' if round(state[:, :, P-2].mean()) == 1 else ''
		QB = 'q' if round(state[:, :, P-1].mean()) == 1 else ''
		if all(castle=='' for castle in [KW, QW, KB, QB]): QB = '-'
		
		game = '/'.join(game)
		game += " " + turn + " " + KW + QW + KB + QB + " - 0 1"  #ignore turn count rules for now

		return ChessGame(game)

	def __str__(self):
		return str(self.game)

	def __repr__(self):
		return repr(self.game)




if __name__ == "__main__":
	
	game = ChessGame()
	game = ChessGame.load(game.state())
	print (game)
	print (repr(game))
	print ()

	for i in range(0, 100):
		actions = list(game.valid_moves())
		game.make_move(random.choice(actions))

		game = ChessGame.load(game.state())

		print (game)
		print (repr(game))
		print ()


