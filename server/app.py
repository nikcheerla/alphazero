
from flask import Flask
from flask import request

import torch
from chess import Board
from alphachess import ChessNet, MCTSController
from multiprocessing import Pool, Manager


app = Flask(__name__)

manager = Manager()
model = ChessNet()
model.compile(torch.optim.Adadelta, lr=0.3)
controller = MCTSController(manager, model)


@app.route('/<path:path>')
def choose_move(path):
	game = Board(request.args.get('fen'))
	print (game)
	game.push(controller.best_move(game, playouts=2))
	return game.fen()
	

if __name__ == "__main__":
    app.run(host='0.0.0.0')

