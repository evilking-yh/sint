import numpy as np

from entities.function import Function


class L2Loss(Function):

    def forward(self, predict, label):
        return np.sum(0.5 * (predict - label) ** 2)

    def backward(self, output, predict, label):
        delta = predict - label
        return [delta, -delta]
