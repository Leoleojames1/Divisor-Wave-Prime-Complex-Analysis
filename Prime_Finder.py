import math
import numpy as np
from decimal import Decimal, getcontext

def b(x):
    getcontext().prec = 10 # Set the precision of Decimal to 100 decimal places
    result = abs(np.prod([Decimal(x) / n * (Decimal(x) * Decimal(math.pi)) * np.prod(
        [Decimal(1) - Decimal(x) ** Decimal(2) / (Decimal(n) ** 2 * Decimal(k) ** 2) for k in range(2, int(x) + 1)]
    ) for n in range(2, int(x) + 1)]))
    return result

def c(x):
    if b(x) != 0:
        return x
    else:
        return None

x_values = []
y_values = []

x = 2
while True:
    cx = c(x)
    if cx is not None:
        print("c({}) = {}".format(x, cx))

    x += 1
    # if x > 1000:
    #      break