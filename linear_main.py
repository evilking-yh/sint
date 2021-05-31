import numpy as np

from entities.loss import L2Loss
from entities.tensor import Tensor
from module.linear import Linear
from optim.optimizer import SGDOptimizer

epoch_num = 5000
linear = Linear(2, 2)
l2_loss = L2Loss()
optimizer = SGDOptimizer(linear.parameters(), 0.005)

for epoch in range(epoch_num):
    x = np.random.random(2)
    y = -x + 3

    input, label = Tensor(x, require_grad=False), Tensor(y, require_grad=False)
    optimizer.zero_grad()
    output = linear(input)
    loss = l2_loss(output, label)
    loss.backward()
    optimizer.step()

    print("=====================epoch: ", epoch)
    print("loss: ", loss)
    print("weight: ")
    print(linear.weight)
    print("b: ")
    print(linear.b)
