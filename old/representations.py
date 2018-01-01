
import sys, os, random

from itertools import product
import numpy as np
import chess

import IPython


pieces = ["P", "N", "B", "R", "Q", "K"]

def state(game=chess.Board()):

	positions = [line.split(" ") for line in str(game).split("\n")]
	H, W = len(positions), len(positions[0])
	state = np.zeros((H, W, len(pieces)+5))
	for i, j in product(range(H), range(W)):
		if positions[i][j].upper() in pieces:
			# White pieces are +1, black pieces are -1
			state[i, j, pieces.index(positions[i][j].upper())] = +1 if positions[i][j].isupper() else -1
	turn = 1 if game.turn else 0
	KW, KB = game.has_kingside_castling_rights(0), game.has_kingside_castling_rights(1)
	QW, QB = game.has_queenside_castling_rights(0), game.has_queenside_castling_rights(1)
	state[:, :, len(pieces)].fill(turn)
	state[:, :, len(pieces)+1].fill(1 if KW else 0)
	state[:, :, len(pieces)+2].fill(1 if QW else 0)
	state[:, :, len(pieces)+3].fill(1 if KB else 0)
	state[:, :, len(pieces)+4].fill(1 if QB else 0)

	return state

def board(state=state(chess.Board())):

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

	return chess.Board(game)



if __name__ == "__main__":
	
	game = chess.Board()
	game = board(state(game))
	print (game)
	print (repr(game))
	print ()

	for i in range(0, 100):
		actions = list(game.generate_legal_moves())
		game.push(random.choice(actions))

		game = board(state(game))

		print (game)
		print (repr(game))
		print ()


