from entities.function import AddFunction, MatMulFunction
from entities.parameter import Parameter
from module.module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = Parameter.ones((in_features, out_features))
        self.b = Parameter.ones(out_features)

        self.mul_function = MatMulFunction()
        self.add_function = AddFunction()

    def forward(self, x):
        out = self.mul_function(x, self.weight)
        out = self.add_function(out, self.b)
        return out
