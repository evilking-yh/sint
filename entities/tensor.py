import numpy as np


class Tensor(np.ndarray):

    def __new__(cls, data, require_grad=True):
        self = np.asarray(data, dtype=np.float).view(cls)
        self.step_grad = []
        self.grad = None
        self.require_grad = require_grad
        self.inputs = []
        ...
        return self

    def __getattr__(self, item):
        if item not in self.__dict__:
            self.__update_attribute()
        return self.__dict__[item]

    def __update_attribute(self):
        self.step_grad = self.__dict__.get("step_grad", [])
        self.grad = self.__dict__.get("grad", None)
        self.require_grad = self.__dict__.get('require_grad', True)
        self.inputs = self.__dict__.get('inputs', [])

    @classmethod
    def ones(cls, shape, require_grad=True):
        return cls(np.ones(shape), require_grad)

    @classmethod
    def zeros(cls, shape, require_grad=True):
        return cls(np.zeros(shape), require_grad)

    @classmethod
    def random(cls, shape, require_grad=True):
        return cls(np.random.random(shape), require_grad)

    @classmethod
    def index_vector(cls, index, size, require_grad=True):
        if index >= size:
            raise ValueError("index out of size.")
        data = [0] * size
        data[index] = 1
        return cls(data, require_grad)

    def matmul(self, weight):
        return Tensor(np.matmul(self, weight))

    def sin(self):
        return np.sin(self)

    def cos(self):
        return np.cos(self)

    def fmax(self, other=0):
        return np.fmax(self, other)

    def fmin(self, other=0):
        return np.fmin(self, other)

    def backward(self):
        if self.grad is None:
            self.grad = 1
        for sg, input in zip(self.step_grad, self.inputs):
            if not input.require_grad:
                continue
            input.grad = self.grad * sg
            input.backward()
