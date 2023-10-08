"""
Prime_Finder_Product_Script:
    This script was designed to calculate the prime numbers up to a given x value using the infinite
    product of the infinite product representation of sin(pi*x/n) through the use of a piecewise function.
4/9/2023
@LeoBorcherding
"""
import math

import numpy as np
import sympy
import scipy
from scipy import special

def sieve_double_product(x):
    """
    Computes the product of the product representation for sin(z).
    ∏_(n=2)^x (pi*x) ∏_(n=2)^x (1-(x^2)/(i^2)(n^2))

    Args:
        x (Real): A real, whole number to evaluate.
    Returns:
        (sympy.Rational): The product of the product representation for sin(z) as an exact fraction.
    """
    result = abs(sympy.prod([
        x / n * (x * sympy.pi) * sympy.prod([
            1 - (x ** 2) / (n ** 2 * k ** 2)
            for k in range(2, int(x) + 1)
        ])
        for n in range(2, int(x) + 1)
    ]))
    return result

def zeta_double_product(x):
    """
    Computes the product of the product representation for sin(z).


    Args:
        x (Real): A real, whole number to evaluate.
    Returns:
        (sympy.Rational): The product of the product representation for sin(z) as an exact fraction.
    """
    result = abs(sympy.prod([
        (2 ** x) * (sympy.pi ** (x - 1)) * sympy.sin((x * sympy.pi)/sympy.S(n)) * sympy.gamma(x - 1) * sympy.prod([
            1 - (sympy.S(n) ** (1 - x)) / (k ** (1 - x) * sympy.S(n) ** (1 - x))
            for k in range(2, int(x) + 1) ])
        for n in range(2, int(x) + 1) ]))
    return result

def zeta_product(x):
    """
    Computes the product of the product representation for sin(z).


    Args:
        x (Real): A real, whole number to evaluate.
    Returns:
        (sympy.Rational): The product of the product representation for sin(z) as an exact fraction.
    """
    result = sympy.Rational(abs(
        (2 ** x) * (sympy.pi ** (x - 1)) * sympy.sin((x * sympy.pi)/2) * sympy.gamma(x - 1) * sympy.prod([
            1 - (2 ** (1 - x)) / (k ** (1 - x))
            for k in range(2, int(x) + 1)
        ])))
    return result

def magic_product(x):
    """
    Computes the product of the product representation for sin(z).

    Args:
        x (Real): A real, whole number to evaluate.
    Returns:
        (sympy.Rational): The product of the product representation for sin(z) as an exact fraction.
    """
    euler_mascheroni = 0.57721566490153286060651209008240243104215933593992359880
    # Product of double product representation of Sin using gamma
    result = \
        sympy.expand(abs(sympy.prod([((-sympy.pi * k)/x) *

            ((x / k) * (sympy.exp(sympy.EulerGamma * x / k)) *
            sympy.prod([ (1 + x/(n*k)) * sympy.exp(-x/n*k)
            for n in range(2, int(x) + 1)]) ) ** (1) *

            (((-x) / k) * (sympy.exp(sympy.EulerGamma * (-x) / k)) *
            sympy.prod([ (1 - x/(n*k)) * sympy.exp(x/n*k)
            for n in range(2, int(x) + 1)]) ) ** (1)

        for k in range(2, int(x) + 1)])))

    # result = \
    #     abs(sympy.prod([((-math.pi * k)/x) * (scipy.special.gamma(x/k) ** (-1)) * (scipy.special.gamma(-x/k) ** (-1))
    #         for k in range(2, int(x) + 1)]))

    return result

def not_equals_zero_sieve(x):
    """
    Returns x if b(x) is not equal to 0, else returns None

    Args:
        x (Real): A real, whole number to evaluate.
    Returns:
        prime number values of x
    """
    if sieve_double_product(x) != 0:
        return x
    else:
        return None

if __name__ == '__main__':
    # the first number to check for primality
    start = 2
    # the last number to check for primality plus one
    end = 10
    # loop through values of x and form list
    for x in range(start, end):
        # not_equals_zero_sieve_x = not_equals_zero_sieve(x)
        # sieve_double_product_x = sieve_double_product(x)
        # zeta_double_product_x = zeta_double_product(x)
        magic_product_x = magic_product(x)
        # if magic_product_x != 0:
        #     print(f"\nf({x}) = \n")
            #sympy.pprint(sympy.nsimplify(sieve_double_product_x)) # use sympy.pprint() to display the fraction form of the result
            # sympy.pprint(sympy.nsimplify(zeta_double_product_x))
        sympy.pprint(sympy.nsimplify(magic_product_x))