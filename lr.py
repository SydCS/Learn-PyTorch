import numpy as np
import matplotlib.pyplot as plt


def lambda_lr(s):
    warm_up = 10000
    s += 1
    return (512 ** -.5) * np.minimum(s ** -.5, s * warm_up ** -1.5)


s = np.linspace(0, 200000, 1000)
y = lambda_lr(s)

plt.plot(s, y)
plt.show()
