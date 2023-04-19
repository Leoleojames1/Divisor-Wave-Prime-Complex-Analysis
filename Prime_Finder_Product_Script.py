"""
Prime_Finder_Product_Script:
    This script was designed to calculate the prime numbers up to a given x value using the infinite
    product of the infinite product representation of sin(pi*x/n) through the use of a piecewise function.
4/9/2023
@LeoBorcherding
"""

import math
import numpy as np
import scipy
from scipy import special

def b(x):
    """
    Computes the product of the product representation for sin(z).
    ∏_(n=2)^x (pi*x) ∏_(n=2)^x (1-(x^2)/(i^2)(n^2))

    Args:
        x (Real): A real, whole number to evaluate.
    Returns:
        (int): The product of the product representation for sin(z).
    """
    result = abs(np.prod([x / n * (x * math.pi) * np.prod(
        [1 - x ** math.pi / (n ** math.pi * k ** math.pi) for k in range(2, int(x) + 1)]
    ) for n in range(2, int(x) + 1)]))
    return result

def c(x):
    """
    Returns x if b(x) is not equal to 0, else returns None

    Args:
        x (Real): A real, whole number to evaluate.
    Returns:
        prime number values of x
    """
    if b(x) != 0:
        return x
    else:
        return None

if __name__ == '__main__':
    # the first number to check for primality
    start = 2
    # the last number to check for primality plus one
    end = 100
    # list for text file
    results = []
    # loop through values of x and form list
    for x in range(start, end):
        cx = c(x)
        bx = b(x)
        #if cx is not None:
            #print("c({}) = {}".format(x, cx))
        print("b({}) = {}".format(x, bx))
    #         results.append("c({}) = {}\n".format(x, cx))
    # # write list to file
    # with open("prime_list_output.txt", "w") as f:
    #     f.writelines(results)