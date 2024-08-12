import numpy as np

from matplotlib import pyplot as plt

if __name__ == '__main__':
    temp = np.random.beta(300, 18, 10000)

    plt.hist(temp)
    plt.show()
