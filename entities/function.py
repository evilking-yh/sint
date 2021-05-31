import numpy as np


class Function(object):

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, output, *inputs):
        return []

    def __call__(self, *inputs):
        t = self.forward(*inputs)
        t.step_grad = self.backward(t, *inputs)
        t.inputs = list(inputs)
        return t


class AddFunction(Function):

    def forward(self, *inputs):
        rst = 0
        for t in inputs:
            rst += t
        return rst

    def backward(self, output, *inputs):
        return [1 for _ in range(len(inputs))]


class MulFunction(Function):

    def forward(self, input, weight):
        return input * weight

    def backward(self, output, input, weight):
        return [weight, input]


class MatMulFunction(Function):

    def forward(self, input, weight):
        return np.dot(input, weight)

    def backward(self, output, input, weight):
        return [weight, input]


class ReluFunction(Function):

    def forward(self, input):
        return input.fmax()

    def backward(self, output, input):
        return (input > 0).astype(int)


class SigmoidFunction(Function):
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, output, input):
        return input * (1 - input)
