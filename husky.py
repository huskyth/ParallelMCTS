from collections import deque
from functools import reduce

import numpy as np

from matplotlib import pyplot as plt

if __name__ == '__main__':
    # temp = np.random.beta(300, 18, 10000)
    #
    # plt.hist(temp)
    # plt.show()
    examples_buffer = deque([], maxlen=2)
    examples_buffer += [(1, 2, 3), (4, 5, 6)]
    examples_buffer += [(10, 20, 30)]
    examples_buffer += [(-1, -2, -3)]
    print(examples_buffer)
    print([(1, 2, 3), (4, 5, 6)] + [(10, 20, 30)] + [(-1, -2, -3)])
    y = reduce(lambda a, b: a + b, examples_buffer)
    print(y)
