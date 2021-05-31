from entities.tensor import Tensor


class Parameter(Tensor):

    def __new__(cls, data, require_grad=True):
        self = Tensor(data, require_grad).view(cls)
        return self
