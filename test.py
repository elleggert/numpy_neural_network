import unittest

from part1_nn_lib import *

linear = LinearLayer(2, 3)
x = np.ones([4,2])
linear.forward(x)
grad_z = np.ones([4,3])
linear.backward(grad_z)
learning_rate = 1
linear.update_params(learning_rate)
