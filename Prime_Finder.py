import math

def b(x):
    prod = 1
    for n in range(2, x+1):
        term = math.pi * x
        for k in range(2, x+1):
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

x = 2
while True:
    cx = c(x)
    if cx is not None:
        print("c({}) = {}".format(x, cx))
        # print("P({}) = {}".format(x, P(x)))
    x += 1
    # if x > 100:
    #     break