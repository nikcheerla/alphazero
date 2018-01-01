
import sys, os, random, time

from itertools import product
import numpy as np
import IPython

from modules import TrainableModel
from games.gomoku import GomokuGame

import multiprocessing
from multiprocessing import Pool, Manager

from mcts import MCTSController
from alphazero import AlphaZeroController

import torch
import torch.nn as nn
import torch.nn.functional as F



""" CNN representing estimated value for each board state.
"""

class Net(TrainableModel):

	def __init__(self):

		super(TrainableModel, self).__init__()
		self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), padding=(1, 1))
		self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
		self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
		self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
		self.layer1 = nn.Linear(128, 256)
		self.layer2 = nn.Linear(256, 1)

	def loss(self, data, data_pred):
		Y_pred = data_pred["target"]
		Y_target = data["target"]
		#print ((Y_pred, Y_target))

		return (F.mse_loss(Y_pred, Y_target))

	def forward(self, x):
		x = x['input']
		x = x.view(-1, 2, 19, 19)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, (2, 2))
		x = F.relu(self.conv3(x))
		x = F.max_pool2d(x, (2, 2))
		x = F.relu(self.conv4(x))

		x = F.max_pool2d(x, (4, 4))[:, :, 0, 0]
		x = F.dropout(x, p=0.2, training=self.training)
		
		x = self.layer1(x)
		x = self.layer2(F.tanh(x))

		return {'target': x}


if __name__ == "__main__":

	manager = Manager()
	model = Net()
	model.compile(torch.optim.Adadelta, lr=0.3)
	controller = MCTSController(manager, T=0.2)

	for i in range(0, 1000):
		game = GomokuGame()
		game.make_move(random.choice(game.valid_moves()))
		print (game)
		print ()

		while not game.over():
			game.make_move(controller.best_move(game, playouts=20))

			print (game)
			print ()

