
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import os, sys, random
from data import torchify, numpify, batch, stack, cast

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import IPython




""" Model that implements the fit_on_batch() method with "compilation" and a custom defined loss function.
Allows a regularizer loss as well as a training loss (based on X, Y data pairs).
Exposed methods: predict_on_batch() to make a batch-wise (numpy) prediction, fit_on_batch() to fit on a minibatch.
Overridable methods: loss(), auxilary_loss(), etc
"""

class AbstractModel(nn.Module):

    def __init__(self):
        
        super(AbstractModel, self).__init__()
        self.compiled = False
        self.inputs = ["input"]
        self.targets = ["target"]

    def __call__(self, x):
        return self.forward(x)

    def compile(self, optimizer=None, **kwargs):

        if optimizer is not None:
            self.optimizer_class = optimizer
            self.optimizer_kwargs = kwargs

        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        self.compiled = True
        if torch.cuda.is_available():
            self.cuda()

    # Subclasses: please override for custom loss functions
    # Right now simple categorical loss is the default
    def loss(self, data, data_pred):
        Y_pred = data_pred["target"]
        Y_target = data["target"]

        return F.nll_loss(Y_pred, Y_target)

    # Subclasses: please override for regularization/other penalties
    def auxiliary_loss(self):
        return 0.0

    # Returns batch data with predicted target
    def __consume_batch(self, data, mask_missing=False):

        data_input = {target: data[target] for target in self.inputs}
        data_pred = self.forward(torchify(data_input))
        data = data.copy()

        if mask_missing:
            for target in self.targets:
                if target not in data or data[target] is None: continue
                mask = np.where(data[target] != None)[0]
                if len(mask) == 0: continue
                data[target] = cast(data[target][mask])
                mask = torch.LongTensor(mask)
                if torch.cuda.is_available():
                    mask = mask.cuda()
                data_pred[target] = data_pred[target][mask]

        return data, data_pred

    def predict_on_batch(self, data):
        data, data_pred = self.__consume_batch(data, mask_missing=False)
        return numpify(data_pred)

    def fit_on_batch(self, data):
        self.train()
        self.zero_grad()

        data, data_pred = self.__consume_batch(data, mask_missing=True)
        data = torchify(data)
        loss = self.loss(data, data_pred) + self.auxiliary_loss()
        loss.backward()

        self.optimizer.step()
        self.eval()

        return loss.cpu().data.numpy().mean()

    def forward(self, x):
        raise NotImplementedError()





""" Model that implements various methods for training and prediction on list/generator objects.
The fit() method fits on a generator/list in real time, with an optional validation generator also provided.
The predict() method predicts based on a generator with only train data provided (the targets are ignored or None).
The 
"""

class TrainableModel(AbstractModel):

    def __init__(self):
        
        super(AbstractModel, self).__init__()
        self.compiled = False
        self.inputs = ["input"]
        self.targets = ["target"]

    def __consume_generator(self, data_generator, fit=False, mask_missing=False):

        preds_history = []
        target_history = []

        for data in data_generator:

            if fit: self.fit_on_batch(data)
            self.train(fit)

            data, data_pred = self._AbstractModel__consume_batch(data, mask_missing=mask_missing)
            data = {target: data[target] for target in self.targets}
            data_pred = numpify(data_pred)
            preds_history.append(data_pred)
            target_history.append(data)

        preds_history = stack(preds_history)
        target_history = stack(target_history)

        return preds_history, target_history

    def predict(self, input_generator, batch_size=1):
        input_generator = batch(input_generator, batch_size=batch_size, targets=self.inputs + self.targets)
        preds, targets = self.__consume_generator(input_generator, fit=False, mask_missing=False)
        return preds['target']

    # Subclasses: please override depending on what type of scoring is required
    def score(self, preds, targets):

        score = accuracy_score(np.argmax(preds['target'], axis=1), targets['target'])
        base_score = accuracy_score(np.argmax(preds['target'], axis=1), shuffle(targets['target']))
        return "{0:.4f}/{1:.4f}".format(score, base_score)        

    def fit(self, train_gen, val_gen=None, batch_size=32, verbose=True):

        if not self.compiled: raise Exception("Model not compiled")

        train_gen = batch(train_gen, batch_size=batch_size, targets=self.inputs + self.targets)
        preds, targets = self.__consume_generator(train_gen, fit=True, mask_missing=True)
        if verbose: print ("Train scores: ", self.score(preds, targets))

        if val_gen is not None:
            val_gen = batch(val_gen, batch_size=batch_size, targets=self.inputs + self.targets)
            preds, targets = self.__consume_generator(val_gen, fit=False, mask_missing=True)
            if verbose: print ("Validation scores: ", self.score(preds, targets))

        self.checkpoint()

    def evaluate(self, test_gen, batch_size=1):

        test_gen = batch(test_gen, batch_size=batch_size, targets=self.inputs + self.targets)
        preds, targets = self.__consume_generator(test_gen, fit=False, mask_missing=True)
        print ("Evaluation scores: ", self.score(preds, targets))

    # Subclasses: please override to extend functionality
    def checkpoint(self):
        pass

