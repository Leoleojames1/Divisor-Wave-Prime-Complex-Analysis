import math
# import matplotlib.pyplot as plt
# import numpy as np


def b(x):
    prod = 1
    for n in range(2, int(x)+1):
        term = math.pi * x
        for k in range(2, int(x)+1):
            term *= (1 - (x**2) / ((k**2) * (n**2)))
        prod *= term
    return prod

def c(x):
    if b(x) != 0:
        return x
    else:
        return None

def P(x):
    if b(x) != 0 and x > 0:
        prod = 1
        for n in range(2, x+1):
            prod *= math.sin(math.pi * x / c(x))
        return prod
    else:
        return 1

x_values = []
y_values = []
# x_values = np.arange(0.2, 10.0, 0.1)
# c_values = np.array([])

x = 2
while True:
# for x in x_values:
    cx = c(x)
    if cx is not None:
        # c_values = np.append(c_values, cx)

        # TODO Print the prime numbers in the terminal, un-comment
        print("c({}) = {}".format(x, cx))
        # print("P({}) = {}".format(x, P(x)))

    x += 1
    if x > 100:
         break

# plt.plot(x_values, c_values)
# plt.title('Graph of P(x)')
# plt.xlabel('x')
# plt.ylabel('P(x)')
# plt.show()