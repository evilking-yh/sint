class Optimizer(object):

    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError("`step` method is not implemented.")

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = None
                param.step_grad = []
                param.inputs = []


class SGDOptimizer(Optimizer):

    def __init__(self, params, lr):
        super(SGDOptimizer, self).__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is None:
                raise ValueError("Grad has not been calculated for Node: {}".format(param.name))
            param -= self.lr * param.grad
