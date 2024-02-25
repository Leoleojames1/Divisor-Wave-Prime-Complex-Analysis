"""
Special_Functions.py
    The Special functions class is a set of special functions in the complex plane involving infinite products in the
    complex plane. These products are related to the distribution of prime numbers and the Riemann Hypothesis. The
    methods in this class are called by the file Complex_Plotting.py for complex the creation of complex plots.
4/9/2023
@LeoBorcherding
"""

import cmath
import math
import os
import pprint

import mpmath
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import special
from scipy import constants
import re

plt.style.use('dark_background')

# -----------------------------------------------------------------------------------------------------------------
class Special_Functions :
    """
    Special_Functions is a class designed to organize the many complex functions that may want to be graphed with the
    Complex_Plotting Class.
    """
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, plot_type):
        """
        Initialization the Special_Functions Object, defines self arg m = desired magnification exponent
        """

        # Regex for identifying product symbols TODO Implement Summation Factory Function
        self.pattern = r'([a-z]+)_\(([a-z]+)=([0-9]+)\)\^\(([a-z]+)\=([0-9]+)\)\s*\[(.*)\]'

        return

    # -----------------------------------------------------------------------------------------------------------------------
    # Product of Sin(x) via Sin(x) product Representation.
    def product_of_sin(self, z, Normalize_type):
        """
        Computes the infinite product of sin(pi*z/n)
        f(x) = ∏_(n=2)^x { abs ( sin(pi*x/n) ) ] ) }
        Args:
            z (complex): A complex number to evaluate.
        Returns:
            (float): The product of sin(pi*z/n).
        """
        # initialize complex components
        z_real = np.real(z)
        z_imag = np.imag(z)
        # initialize scaling coefficients

        if Normalize_type == 'Y':
            # self.m = 0.0125
            # self.beta = 0.0125
            # self.m = 0.005125
            # self.beta = 0.006125
            self.m = 0.0465
            self.beta = 0.178
        else:
            # self.m = 0.0125
            # self.m = 0.005125
            # self.beta = 0.006125
            self.m = 0.0465
            self.beta = 0.178

        # calculate infinite product
        result = abs(np.prod(
            [self.beta * ((z_real) / k) * np.sin(math.pi * (z_real + 1j * z_imag) / k)
                for k in range(2, int(z_real) + 1)])) ** (-self.m)

        if Normalize_type == 'Y':
            result = result / scipy.special.gamma(result)
        else:
            result = result

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def product_of_product_representation_for_sin(self, z, Normalize_type):
        """
        Computes the product of the product representation for sin(pi*z/n).
        f(x) = ∏_(n=2)^x (pi*x) ∏_(n=2)^x (1-(x^2)/(i^2)(n^2))
        Args:
            z (complex): A complex number to evaluate.
        Returns:
            (float): The product of the product representation for sin(pi*z/n).
        """

        #pprint.pprint(z)

        #TODO no magnification for bug testing
        z_real = np.real(z)
        z_imag = np.imag(z)

        # pprint.pprint(z_real)
        # pprint.pprint(z_imag)

        #27 is a fun exponent to try
        power_integer = 2

        if Normalize_type == 'Y':
            # self.m = 0.0125
            # self.m = 0.37
            # # m = 0.07
            # # m = 0.37
            # self.beta = 0.078
            self.m = 0.0125
            self.beta = 0.078
        else:
            # self.m = 0.001
            # # self.m = 0.007
            # self.m = 0.0125
            # # self.m = 0.37
            # self.beta = 0.078
            self.m = 0.0125
            self.beta = 0.078

        # calculate the double infinite product via the double for loop
        result = abs(np.prod(
            [self.beta * ( z_real / n) * (((z_real + 1j * z_imag) * math.pi) * np.prod(
                [1 - ( (z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)

        # # print function values for bug testing
        # if (z_real % 1 == 0) and (z_imag % 1 == 0):
        #     print(z, ": ", result)

        if Normalize_type == 'Y':
            result = result / scipy.special.gamma(result)
        else:
            result = result

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def product_of_product_representation_for_sin_COMPLEX_VARIANT(self, z, Normalize_type):
        """
        Computes the product of the product representation for sin(pi*z/n).
        f(x) = ∏_(n=2)^x (pi*x) ∏_(n=2)^x (1-(x^2)/(i^2)(n^2))
        Args:
            z (complex): A complex number to evaluate.
        Returns:
            (float): The product of the product representation for sin(pi*z/n).
        """

        #pprint.pprint(z)

        #TODO no magnification for bug testing
        z_real = np.real(z)
        z_imag = np.imag(z)

        # pprint.pprint(z_real)
        # pprint.pprint(z_imag)

        #27 is a fun exponent to try
        power_integer = 2

        if Normalize_type == 'Y':
            self.m = 0.0125
            self.beta = 0.054
        else:
            self.m = 0.0125
            self.beta = 0.054

        # calculate the double infinite product via the double for loop
        result = abs(np.prod(
            [self.beta * ( (1j * z_imag) / n) * (((z_real + 1j * z_imag) * math.pi) * np.prod(
                [1 - ( (1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int((1j * z_imag)) + 1)])) for n in range(2, ((1j * z_imag)) + 1)])) ** (-self.m)

        prime_sieve = abs(np.prod(
            [self.beta * ((1j * z_imag) / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ( (1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int((1j * z_imag)) + 1)])) for n in range(2, ((1j * z_imag)) + 1)])) ** (-self.m)

        # # print function values for bug testing
        # if (z_real % 1 == 0) and (z_imag % 1 == 0):
        #     print(z, ": ", result)

        result = prime_sieve ** result

        if Normalize_type == 'Y':
            result = result / scipy.special.gamma(result)
        else:
            result = result

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def natural_logarithm_of_product_of_product_representation_for_sin(self, z, Normalize_type):
        """
        Computes the natural logarithm of the normalized infinite product of product representation of sin(z).
        f(x) = ln ( ∏_(n=2)^x (pi*x) ∏_(n=2)^x (1-(x^2)/(i^2)(n^2)) )
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (complex): The natural logarithm of the normalized infinite product of product representation of sin(z).
        """
        z_real = np.real(z)
        z_imag = np.imag(z)

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        # calculate the infinite product
        result = abs(np.prod(
            [self.beta * (z_real * imaginary_magnification / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)

        # take the logarithm
        log_result = cmath.log(result)

        #log_result=cmath.log(result) # scipy.special.gamma(
        #return result / np.log(z)
        return log_result

    # -----------------------------------------------------------------------------------------------------------------
    def gamma_of_product_of_product_representation_for_sin(self, z, Normalize_type):
        """
        Args:
        Returns:
        """
        z_real = np.real(z)
        z_imag = np.imag(z)

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        result = abs(np.prod(
            [self.beta * (z_real * imaginary_magnification / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        gamma_result = scipy.special.gamma(result)
        #log_result=cmath.log(result) # scipy.special.gamma(
        return gamma_result

    # -----------------------------------------------------------------------------------------------------------------
    def factored_form_for_product_of_product_representation_for_sin(self, z):
        """
        Computes the product of the product representation for sin(z).
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (float): The product of the product representation for sin(z).
        """
        z_real = np.real(z)
        z_imag = np.imag(z)
        result = abs(np.prod(
            [self.beta * (z_imag * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [(k*n - (z_real + 1j * z_imag))*(k*n + (z_real + 1j * z_imag)) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def gamma_form_product_of_product_representation_for_sin(self, z, Normalize_type):
        """
        Computes the product of the product representation for sin(z).
         z / [Γ(z) * Γ(1-z)] * e^(-γz^2) * ∏_(n=2)^(∞) e^(z^2 / n^2)
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (float): The product of the product representation for sin(z).
        """
        Euler_mascheroni = 0.57721566490153286060651209008240243104215933593992359880
        z_real = np.real(z)
        z_imag = np.imag(z)
        result = abs(np.prod(
            [self.beta * (z_imag * z_real / n) * scipy.special.gamma(z_real + 1j * z_imag)
                * scipy.special.gamma(1 - (z_real + 1j * z_imag))
                    * cmath.exp((-((z_real + 1j * z_imag) ** 2))) * np.prod(
                        [(cmath.exp(((z_real + 1j * z_imag)**2/(n**2)*(k**2))))
                            for k in range(2, int(z_real) + 1)]) for n in range(2, int(z_real) + 1)])) ** (-self.m)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def infinite_product_representation_of_the_zeta_function(self, z, Normalize_type):
        """
        Computes the product of the product representation for sin(z).
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (float): The product of the product representation for sin(z).
        """
        z_real = np.real(z)
        #z_real = 1/2
        z_imag = np.imag(z)
        result = \
            abs(self.beta * (z_imag * z_real)*(2**(z_real + 1j * z_imag) * math.pi**((z_real + 1j * z_imag) - 1)
                * np.sin(math.pi * (z_real + 1j * z_imag) / 2) * scipy.special.gamma(1 - (z_real + 1j * z_imag))
                    * np.prod([(2**(1-(z_real + 1j * z_imag))) / (k**(1 - (z_real + 1j * z_imag)))
                        for k in range(1, int(z_real) + 1)]))) ** (-self.m)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def product_of_infinite_product_representation_of_the_zeta_function(self, z, Normalize_type):
        """
        Computes the product of the product representation for sin(z).
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (float): The product of the product representation for sin(z).
        """
        z_real = np.real(z)
        #z_real = 1/2
        z_imag = np.imag(z)
        result = \
            abs(np.prod([(1j * z_imag * z_real/n)*(2**(z_real + 1j * z_imag) * math.pi**((z_real + 1j * z_imag) - 1)
                * np.sin(math.pi * (z_real + 1j * z_imag) / 2) * scipy.special.gamma(1 - (z_real + 1j * z_imag))
                    * np.prod([(2**(1-(z_real + 1j * z_imag))) / (k**(1 - (z_real + 1j * z_imag))) *
                        (n**(1 - (z_real + 1j * z_imag)))
                            for k in range(1, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def normalized_product_of_infinite_product_representation_of_the_zeta_function(self, z):
        """
        Computes the product of the product representation for sin(z).
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (float): The product of the product representation for sin(z).
        """
        z_real = np.real(z)
        # z_real = 1/2
        z_imag = np.imag(z)

        numerator = \
            abs(np.prod([(1j * z_imag * z_real/n)*(2**(z_real + 1j * z_imag) * math.pi**((z_real + 1j * z_imag) - 1)
                * np.sin(math.pi * (z_real + 1j * z_imag) / 2) * scipy.special.gamma(1 - (z_real + 1j * z_imag))
                    * np.prod([(2**(1-(z_real + 1j * z_imag))) / (k**(1 - (z_real + 1j * z_imag))) *
                        (n**(1 - (z_real + 1j * z_imag)))
                            for k in range(1, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)

        denominator = \
            abs(np.prod([(1j * z_imag * z_real/n)*(2**(z_real + 1j * z_imag) * math.pi**((z_real + 1j * z_imag) - 1)
                * np.sin(math.pi * (z_real + 1j * z_imag) / 2) * scipy.special.gamma(1 - (z_real + 1j * z_imag))
                    * np.prod([(2**(1-(z_real + 1j * z_imag))) / (k**(1 - (z_real + 1j * z_imag))) *
                        (n**(1 - (z_real + 1j * z_imag)))
                            for k in range(1, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)

        normalized_fraction = (numerator ** (-self.m) / scipy.special.gamma(denominator ** (-self.m)))

        return normalized_fraction

    # -----------------------------------------------------------------------------------------------------------------
    def product_combination(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """
        # result = \
        #     self.product_of_sin(z) \
        #     * self.product_of_product_representation_for_sin(z)

        # result = \
        #     self.normalized_product_of_sin(z) \
        #     * self.normalized_product_of_product_representation_for_sin(z)

        # if 1/result(z) != 0:
        #     return z
        # else:
        #     return None

        # result = \
        #     self.product_of_product_representation_for_sin(z) \
        #     * self.normalized_product_of_product_representation_for_sin(z)

        # combination = \
        #     self.product_of_product_representation_for_sin(z) \
        #     * math.e ** (z)

        # z_real = np.real(z)
        # z_imag = np.imag(z)
        #
        # # normalized double infinite product for loop
        # combination = np.cos(abs(np.prod([self.beta * (z_real / n) * (
        #         (z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
        #     [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
        #                          range(2, int(z_real) + 1)])))

        # denominator = abs(np.prod([self.beta * (z_real / n) * (
        #         (z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
        #     [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
        #                            range(2, int(z_real) + 1)]))
        # normalized_fraction = (numerator ** (-self.m) / scipy.special.gamma(denominator ** (-self.m)))

        # combination = np.cos(normalized_fraction)

        # combination = \
        #     self.product_of_product_representation_for_sin(z) * self.Riesz_Product_for_Sin(z) * self.Viete_Product_for_Sin(z)
        #


        #TODO ADD THIS AS THE BINARY OUTPUT SEIVE
        z_real = np.real(z)
        z_imag = np.imag(z)

        # calculate numerator of normalized fraction
        FUNC_A = abs(np.prod(
            [self.beta * (z_real * 1 / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        # calculate the double infinite product via the double for loop
        FUNC_B = abs(np.prod(
            [self.beta * (1 * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ( (z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)

        A_NORM = (FUNC_A / scipy.special.gamma(FUNC_A))
        B_NORM = (FUNC_B / scipy.special.gamma(FUNC_B))

        pow_fun = A_NORM ** B_NORM
        # pow_fun = FUNC_A ** FUNC_B

        return pow_fun

    # -----------------------------------------------------------------------------------------------------------------
    def product_of_sin_plus_normalized_product_of_sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        #TODO implement seperate values for beta & m so that functions can be fine tuned as each function has
        # its own magnification scaling

        def f(z):
            """
            product of sin
            """
            z_real = np.real(z)
            z_imag = np.imag(z)
            result = abs(np.prod([4.577 * (z_real / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                                  for n in range(2, int(z_real) + 1)])) ** (-self.m)
            return result

        def g(z):
            """
            normalized product of sin
            """
            z_real = np.real(z)
            z_imag = np.imag(z)
            numerator = abs(np.prod([4.577 * (z_real / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                                     for n in range(2, int(z_real) + 1)])) ** (-self.m)
            denominator = abs(np.prod([4.577 * (z_real / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                                       for n in range(2, int(z_real) + 1)])) ** (-self.m)

            normalized_fraction = (numerator ** (-self.m) / scipy.special.gamma(denominator ** (-self.m)))
            return normalized_fraction

        combination = g(z) + f(z)

        return combination

    # -----------------------------------------------------------------------------------------------------------------
    def cos_of_product_of_sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054
        else:
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054

        # normalized double infinite product for loop
        result = np.cos(abs(np.prod([self.beta * (z_real / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                              for n in range(2, int(z_real) + 1)])) ** (-self.m))

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def sin_of_product_of_sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054
        else:
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054

        # normalized double infinite product for loop
        result = np.sin(abs(np.prod([self.beta * (z_real / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                              for n in range(2, int(z_real) + 1)])) ** (-self.m))

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def cos_of_product_of_product_representation_of_sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054
        else:
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054

        # normalized double infinite product for loop
        result = np.cos(abs(np.prod([self.beta * (z_real / n) * (
                (z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
            [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
                                 range(2, int(z_real) + 1)]))) ** (-self.m)

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def sin_of_product_of_product_representation_of_sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054
        else:
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054

        # normalized double infinite product for loop
        result = np.sin(abs(np.prod([self.beta * (z_real / n) * (
                (z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
            [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
                                 range(2, int(z_real) + 1)]))) ** (-self.m)

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def Riesz_Product_for_Cos(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            m = 0.08
            self.beta = 1
        else:
            m = 0.08
            self.beta = 1

        # calculate infinite product
        num = abs(np.prod(
            [(1 + np.cos(math.pi * (z_real + 1j * z_imag) * n))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        norm = num / scipy.special.gamma(num)

        return norm

    # -----------------------------------------------------------------------------------------------------------------
    def Riesz_Product_for_Sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)
        m = 0.08

        if Normalize_type == 'Y':
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054
        else:
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054

        # calculate infinite product
        num = abs(np.prod(
            [1 + np.sin(math.pi * (z_real + 1j * z_imag) * n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        norm = num / scipy.special.gamma(num)

        return norm

    # -----------------------------------------------------------------------------------------------------------------
    def Riesz_Product_for_Tan(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)
        # m = 0.26
        # m = 0.08

        if Normalize_type == 'Y':
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054
        else:
            # self.m = 1
            # self.beta = 1
            self.m = 0.0125
            self.beta = 0.054

        # calculate infinite product
        num = abs(np.prod(
            [1 + np.tan(math.pi * (z_real + 1j * z_imag) * n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        norm = num / scipy.special.gamma(num)

        return norm

    # -----------------------------------------------------------------------------------------------------------------
    def Viete_Product_for_Cos(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.37
            self.beta = 1
        else:
            self.m = 0.07
            self.beta = 1

        #TODO different values for the base of the denominator try 2, 3, 4, 1/2, phi, pi, e, euler-mascheroni
        # scipy.constants.golden

        # calculate infinite product
        num = abs(np.prod(
            [np.cos(math.pi * (z_real + 1j * z_imag) / (2 ** (n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        #This doesnt really need to start at 2, it could start at 1
        norm = num / scipy.special.gamma(num)

        #TODO add conditional statement for normalization of the function,
        # if user norm yes then return norm else return num

        return norm

    # -----------------------------------------------------------------------------------------------------------------
    def Viete_Product_for_Sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.37
            self.beta = 1
        else:
            self.m = 0.07
            self.beta = 1

        # check for imaginary magnification
        # if self.im_mag is True:
        #     imaginary_magnification = z_imag
        # else:
        #     imaginary_magnification = 1

        #TODO different values for the base of the denominator try 2, 3, 4, 1/2, phi, pi, e, euler-mascheroni
        # scipy.constants.golden

        # calculate infinite product
        num = abs(np.prod(
            [np.sin(math.pi * (z_real + 1j * z_imag) / (2 ** (n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        den = abs(np.prod(
            [np.sin(math.pi * (z_real + 1j * z_imag) / (2 ** (n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        norm = num / scipy.special.gamma(den)

        #TODO add conditional statement for normalization of the function,
        # if user norm yes then return norm else return num

        return norm

    # -----------------------------------------------------------------------------------------------------------------
    def Viete_Product_for_Tan(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.37
            self.beta = 0.07
        else:
            self.m = 0.07
            self.beta = 0.07

        # # check for imaginary magnification
        # if self.im_mag is True:
        #     imaginary_magnification = z_imag
        # else:
        #     imaginary_magnification = 1

        #TODO different values for the base of the denominator try 2, 3, 4, 1/2, phi, pi, e, euler-mascheroni
        # scipy.constants.golden

        # calculate infinite product
        num = abs(np.prod(
            [np.tan(math.pi * (z_real + 1j * z_imag) / (2 ** (n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        den = abs(np.prod(
            [np.tan(math.pi * (z_real + 1j * z_imag) / (2 ** (n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        norm = num / scipy.special.gamma(den)

        #TODO add conditional statement for normalization of the function,
        # if user norm yes then return norm else return num

        return num

    # -----------------------------------------------------------------------------------------------------------------
    def Half_Base_Viete_Product_for_Sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.37
            self.beta = 0.07
        else:
            self.m = 0.07
            self.beta = 0.07

        # # check for imaginary magnification
        # if self.im_mag is True:
        #     imaginary_magnification = z_imag
        # else:
        #     imaginary_magnification = 1

        #TODO different values for the base of the denominator try 2, 3, 4, 1/2, phi, pi, e, euler-mascheroni
        # scipy.constants.golden

        # calculate infinite product
        num = abs(np.prod(
            [np.sin(math.pi * (z_real + 1j * z_imag) / (2 ** (-n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        den = abs(np.prod(
            [np.sin(math.pi * (z_real + 1j * z_imag) / (2 ** (-n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)
        #THis doesnt really need to start at 2, it could start at 1
        norm = num / scipy.special.gamma(den)

        #TODO add conditional statement for normalization of the function,
        # if user norm yes then return norm else return num

        return norm

    # -----------------------------------------------------------------------------------------------------------------
    def Riesz_Product_for_Tan_and_Prime_indicator_combination(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)
        m = 0.26
        q = 0.08

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # calculate infinite product
        num = abs(np.prod(
            [1 + np.tan(math.pi * (z_real + 1j * z_imag) * n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        FUNC_B = abs(np.prod(
            [(1 * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ( (z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-q)

        norm = num / scipy.special.gamma(num)
        FUNC_B_norm = FUNC_B / scipy.special.gamma(FUNC_B)

        # num_exp = FUNC_B_norm ** norm
        # num_exp = norm ** FUNC_B_norm
        # num_exp = norm/FUNC_B_norm
        num_exp = np.cos(FUNC_B_norm/norm)

        return num_exp

    # -----------------------------------------------------------------------------------------------------------------
    def Prime_Indicator_Self_sieve_base(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)
        m = 0.26
        q = 0.08

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # calculate infinite product
        num = abs(np.prod(
            [1 + np.tan(math.pi * (z_real + 1j * z_imag) * n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        FUNC_B = abs(np.prod(
            [(1 * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ( (z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-q)

        norm = num / scipy.special.gamma(num)
        FUNC_B_norm = FUNC_B / scipy.special.gamma(FUNC_B)

        # num_exp = FUNC_B_norm ** norm
        # num_exp = norm ** FUNC_B_norm
        # num_exp = norm/FUNC_B_norm
        num_exp = np.cos(FUNC_B_norm/norm)

        return num_exp

    # -----------------------------------------------------------------------------------------------------------------
    def Nested_roots_product_for_2(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)
        # m = 0.26
        # q = 0.08
        m = -1

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # #TODO Sqrt Sum
        # sum = abs(np.prod(
        #     [ (z_real + 1j * z_imag) ** 2 ** (-n)
        #         for n in range(2, int(z_real) + 1)])) ** (-m)
        # #TODO Sqrt Prod
        # prod = abs(np.sum(
        #     [ (z_real + 1j * z_imag) ** 2 ** (-n)
        #         for n in range(2, int(z_real) + 1)])) ** (-m)

        # #TODO Viete's paradox nest roots sum product representation
        # paradox = abs(np.prod(
        #     [np.sum([( k ** (z_real + 1j * z_imag) ** (-n)) / 2
        #              for k in range(2, int(z_real) + 1)]) for n in range(2, int(z_real) + 1)]))

        #TODO Sqrt Prod
        prod = abs(np.sum(
            [ (z_real + 1j * z_imag) ** 2 ** (-n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)
        paradox = abs(np.prod(
            [np.prod([(z_real + 1j * z_imag) ** k ** (-n)
                     for k in range(2, int(z_real) + 1)]) for n in range(2, int(z_real) + 1)]))

        # norm = num / scipy.special.gamma(num)
        # FUNC_B_norm = FUNC_B / scipy.special.gamma(FUNC_B)

        # num_exp = FUNC_B_norm ** norm
        # num_exp = norm ** FUNC_B_norm
        # num_exp = norm/FUNC_B_norm
        # num_exp = np.cos(FUNC_B_norm/norm)

        return paradox

    # -----------------------------------------------------------------------------------------------------------------
    def Log_power_base_Viete_Product_for_Sin(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        #TODO different values for the base of the denominator try 2, 3, 4, 1/2, phi, pi, e, euler-mascheroni
        # scipy.constants.golden

        # calculate infinite product
        num = 1 + abs(np.prod(
            [np.sin(math.pi * (z_real + 1j * z_imag) / (2 ** ((1/n)*np.log((z_real + 1j * z_imag)))))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        den = 1 + abs(np.prod(
            [np.sin(math.pi * (z_real + 1j * z_imag) / (2 ** ((1/n)*np.log((z_real + 1j * z_imag)))))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)
        #THis doesnt really need to start at 2, it could start at 1
        norm = num / scipy.special.gamma(den)

        #TODO add conditional statement for normalization of the function,
        # if user norm yes then return norm else return num

        return norm

    # -----------------------------------------------------------------------------------------------------------------------
    # Normalized Product of Sin(x) via Sin(x) product Representation, normalized through fraction on gamma function.
    def prime_sieve_0_bz_power(self, z, Normalize_type):
        """
        f(x) = ∏_(n=2)^x beta*pi*n*(∏_(k=2)^x(1-x^2/(k^2*n^2))) / (∏_(n=2)^x(pi*n*(∏_(k=2)^x(1-x^2/(k^2*n^2)))) !
        """
        z_real = np.real(z)
        z_imag = np.imag(z)
        self.m = 0.15
        self.beta = 1.13

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        # calculate numerator of normalized fraction
        numerator = abs(np.prod(
            [self.beta * (z_real * imaginary_magnification / n) * ((z_real * math.pi + 1j * z_imag * math.pi) *
                        np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                    for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)]))

        # calculate denominator of normalized fraction
        denominator = abs(np.prod(
            [self.beta * (z_real * imaginary_magnification / n) * ((z_real * math.pi + 1j * z_imag * math.pi) *
                        np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                    for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)]))

        normalized_fraction = (numerator ** (-self.m) / scipy.special.gamma(denominator ** (-self.m)))
        normalized_fraction = 0 ** normalized_fraction
        # # TODO print function values for bug testing
        # if (z_real % 1 == 0) and (z_imag % 1 == 0):
        #     print(z, ": ", normalized_fraction)

        return normalized_fraction

    # -----------------------------------------------------------------------------------------------------------------
    def Custom_Function(self, z, Normalize_type):
        """
        Computes the product of the product representation for sin(z).
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (float): The product of the product representation for sin(z).
        """
        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        result = abs(np.prod(
            [((1j * z_imag * z_real) / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z_real + 1j * z_imag)**(2)) / (n ** (2) * k ** (2))
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def Custom_Riesz_Product_for_Tan(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        # calculate infinite product
        num = abs(np.prod(
            [1 + np.tan(math.pi * (z_real + 1j * z_imag) * n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        den = abs(np.prod(
            [1 + np.tan(math.pi * (z_real + 1j * z_imag) * n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        norm = num / scipy.special.gamma(den)

        return norm

    # -----------------------------------------------------------------------------------------------------------------
    def Custom_Viete_Product_for_Cos(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        #TODO different values for the base of the denominator try 2, 3, 4, 1/2, phi, pi, e, euler-mascheroni
        # scipy.constants.golden

        # calculate infinite product
        num = abs(np.prod(
            [1 + np.cos(math.pi * (z_real + 1j * z_imag) / (z_real ** (n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        den = abs(np.prod(
            [1 + np.cos(math.pi * (z_real + 1j * z_imag) / (z_real ** (n)))
                for n in range(2, int(z_real) + 1)])) ** (-self.m)
        #THis doesnt really need to start at 2, it could start at 1
        norm = num / scipy.special.gamma(den)

        #TODO add conditional statement for normalization of the function,
        # if user norm yes then return norm else return num

        return norm

    # -----------------------------------------------------------------------------------------------------------------------
    # Product of Sin(x) via Sin(x) product Representation.
    def zeros_of_zeta(self, z, Normalize_type):
        """
        Computes the infinite product of sin(pi*z/n)
        f(x) = ∏_(n=2)^x { abs ( sin(pi*x/n) ) ] ) }
        Args:
            z (complex): A complex number to evaluate.
        Returns:
            (float): The product of sin(pi*z/n).
        """
        # initialize complex components
        z_real = np.real(z) - 1/2
        z_imag = np.imag(z)
        # initialize scaling coefficients
        # self.m = 0.138
        # self.m = 0.158
        # self.m = 0.158
        self.m = 1
        self.beta = 1
        # self.beta = 4.577
        # self.beta = 14.577

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        # # calculate infinite product
        # result = abs(np.prod(
        #     [self.beta * (z_real * imaginary_magnification / n) * np.sin(math.pi * (1j * z_imag) / n)
        #         for n in range(2, int(z_imag) + 1)])) ** (-self.m)
        #
        # self.m = 0.0708
        # self.beta = 0.077

        # # calculate the double infinite product via the double for loop
        # result = abs(np.prod(
        #     [self.beta * ( (z_real + 1j * z_imag) / n) * ((math.pi * (z_real + 1j * z_imag)) * np.prod(
        #         [1 - ( (z_real + 1j * z_imag) ** 2) / (((n*(z_real + 1j * z_imag)) ** 2)*((k*(z_real + 1j * z_imag)) ** 2)*(math.pi ** 2))
        #          for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)

        result = abs(np.prod(
            [self.beta * (z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) *
                        np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                    for k in range(2, int((z_imag)) + 1)])) for n in range(2, int((z_imag)) + 1)]))


        # result = result / scipy.special.gamma(result)

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def Dedekind_Eta_Custom(self, z, Normalize_type):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        # calculate numerator of normalized fraction
        numerator = abs(np.prod(
            [self.beta * (z_real * imaginary_magnification / n) * ((z_real * math.pi + 1j * z_imag * math.pi) *
                        np.prod(
                [1 - ((z_real + 1j * z_imag)) / math.e ** (z * math.pi / (n * k))
                    for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)]))

        # calculate denominator of normalized fraction
        denominator = abs(np.prod(
            [self.beta * (z_real * imaginary_magnification / n) * ((z_real * math.pi + 1j * z_imag * math.pi) *
                        np.prod(
                [1 - ((z_real + 1j * z_imag)) / math.e ** (z * math.pi / (n * k))
                    for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)]))

        normalized_fraction = (numerator ** (-self.m) / scipy.special.gamma(denominator ** (-self.m)))

        return normalized_fraction

    # -----------------------------------------------------------------------------------------------------------------
    def Prime_Binary_Output_Waveform_Product(self, z, Normalize_type):
        """ A method to combine two infinite products to produce a prime binary output.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)
        m = 0.366
        # q = 0.001

        if Normalize_type == 'Y':
            self.m = 0.001
            self.beta = 0.07
        else:
            self.m = 0.001
            self.beta = 0.07

        # calculate infinite product
        single_Prod = abs(np.prod([ np.sin(math.pi * (z_real + 1j * z_imag)/n ) for n in range(2, int(z_real) + 1)]))

        # double_Prod = abs(np.prod([ np.prod([1 - ((z_real + 1j * z_imag)**2)/(k**2 * n**2)]) for k in range(2, int(z_real) + 1) for n in range(2, int(z_real) + 1)]))

        # binary_prime = abs(np.prod(
        #     [ np.sin(math.pi * (z_real + 1j * z_imag)/n )
        #       for n in range(2, int(z_real) + 1)])) ** (abs(np.prod([ np.prod([1 - ((z_real + 1j * z_imag)**2)/(k**2 * n**2)])
        #                                                     for k in range(2, int(z_real) + 1)
        #                                                         for n in range(2, int(z_real) + 1)]))) ** (-m)
        #
        # binary_prime_norm = binary_prime / scipy.special.gamma(binary_prime)

        double_Prod = abs(np.prod([ np.prod([1 - ((z_real + 1j * z_imag)**2)/(k**2 * n**2)]) for k in range(2, int(z_real) + 1) for n in range(2, int(z_real) + 1)]))

        binary_p = np.prod([(1)/(((1 - j * (0 ** 0 ** double_Prod)) for j in range(2, int(z_real) + 1))**(z_real + 1j * z_imag))])
        # binary_p = np.prod([(1)/(((1 - j * (single_Prod ** single_Prod ** double_Prod)) for j in range(2, int(z_real) + 1))**(z_real + 1j * z_imag))])

        return binary_p

    # -----------------------------------------------------------------------------------------------------------------
    def product_factory(self, product_string_arg):
        """
        Args:
            product_string_arg: regex Collected Function
        Returns:
             product_function_from_string: The function converted to code
        """
        match = re.search(self.pattern, product_string_arg)
        if match is None:
            raise ValueError("Invalid formula")

        var1, start1, var2, end2, prod = match.groups()
        start1, end2 = int(start1), int(end2)

        # Create New Product From User String
        def product_function_from_string(z, m):
            z_real = np.real(z)
            z_imag = np.imag(z)
            prod_with_z = prod.replace('z', '(z_real + 1j*z_imag)')
            result = abs(np.prod(
                [(z_imag * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                    [eval(prod_with_z) for k in range(start1, end2 + 1)])) for n in range(start1, end2 + 1)])) ** (-m)
            return result

        # Return the Newly created Product Function
        return product_function_from_string

    # -----------------------------------------------------------------------------------------------------------------
    def lamda_function_library(self, Normalize_type):
        """
        Args:
            None, will eventually be user_input
        Colorization:
        """
        # The function product_of_product_representation_for_sin and Custom_Function are identical however
        # Custom_Function is multiplied by an imaginary scalar for magnification
        # TODO fix the product_factory function, currently broken but will allow users to design their own functions

        # lamda library
        operations = {

            # Library Selection for Complex Analysis of Holomorphic & Meromorphic functions:
            # Todo Special Infinite Products section

            '1': lambda z: self.product_of_sin(z, Normalize_type),
            '2': lambda z: self.product_of_product_representation_for_sin(z, Normalize_type),

            '3': lambda z: self.Riesz_Product_for_Cos(z, Normalize_type),
            '4': lambda z: self.Riesz_Product_for_Sin(z, Normalize_type),
            '5': lambda z: self.Riesz_Product_for_Tan(z, Normalize_type),

            '6': lambda z: self.Viete_Product_for_Cos(z, Normalize_type),
            '7': lambda z: self.Viete_Product_for_Sin(z, Normalize_type),
            '8': lambda z: self.Viete_Product_for_Tan(z, Normalize_type),

            '9': lambda z: self.natural_logarithm_of_product_of_product_representation_for_sin(z, Normalize_type),
            '10': lambda z: self.gamma_of_product_of_product_representation_for_sin(z, Normalize_type),
            '11': lambda z: self.factored_form_for_product_of_product_representation_for_sin(z, Normalize_type),
            '12': lambda z: self.gamma_form_product_of_product_representation_for_sin(z, Normalize_type),

            # Todo Zeta Functions and L functions Section
            '13': lambda z: self.infinite_product_representation_of_the_zeta_function(z, Normalize_type),
            '14': lambda z: self.product_of_infinite_product_representation_of_the_zeta_function(z, Normalize_type),

            # Todo Basic library of functions
            '15': lambda z: abs(mpmath.loggamma(z)),
            '16': lambda z: 1 / (1 + z ^ 2),
            '17': lambda z: abs(z ** z),
            '18': lambda z: mpmath.gamma(z),

            # Todo User Function Utilities
            '19': lambda z: self.Custom_Function(z, Normalize_type),
            '20': lambda z: self.product_factory("∏_(n=2)^z [pi*z ∏_(k=2)^z (1 - z^2 / (k^2 * n^2))]", Normalize_type, ),
            '21': lambda z: self.product_combination(z, Normalize_type),
            '22': lambda z: self.product_of_sin_plus_normalized_product_of_sin(z, Normalize_type),

            '23': lambda z: self.cos_of_product_of_sin(z, Normalize_type),
            '24': lambda z: self.sin_of_product_of_sin(z, Normalize_type),
            '25': lambda z: self.cos_of_product_of_product_representation_of_sin(z, Normalize_type),
            '26': lambda z: self.sin_of_product_of_product_representation_of_sin(z, Normalize_type),

            '27': lambda z: self.Custom_Riesz_Product_for_Tan(z, Normalize_type),
            '28': lambda z: self.Dedekind_Eta_Custom(z, Normalize_type),
            '29': lambda z: self.Custom_Viete_Product_for_Cos(z, Normalize_type),
            '30': lambda z: self.Half_Base_Viete_Product_for_Sin(z, Normalize_type),
            '31': lambda z: self.Log_power_base_Viete_Product_for_Sin(z, Normalize_type),

            '32': lambda z: self.prime_sieve_0_bz_power(z, Normalize_type),
            '33': lambda z: self.zeros_of_zeta(z, Normalize_type),
            '34': lambda z: self.Prime_Binary_Output_Waveform_Product(z, Normalize_type),
            '35': lambda z: self.Riesz_Product_for_Tan_and_Prime_indicator_combination(z, Normalize_type),
            '36': lambda z: self.Nested_roots_product_for_2(z, Normalize_type),
            '37': lambda z: self.product_of_product_representation_for_sin_COMPLEX_VARIANT(z, Normalize_type),
        }

        Catalog = {
            # Todo Special Infinite Products section
            '1': 'product_of_sin(z)',
            '2': 'product_of_product_representation_for_sin(z)',

            '3': 'Riesz_Product_for_Cos(z)',
            '4': 'Riesz_Product_for_Sin(z)',
            '5': 'Riesz_Product_for_Tan(z)',

            '6': 'Viete_Product_for_Cos(z)',
            '7': 'Viete_Product_for_Sin(z)',
            '8': 'Viete_Product_for_Tan(z)',


            '9': 'natural_logarithm_of_product_of_product_representation_for_sin(z)',
            '10': 'gamma_of_product_of_product_representation_for_sin(z)',
            '11': 'factored_form_for_product_of_product_representation_for_sin(z)',
            '12': 'gamma_form_product_of_product_representation_for_sin(z)',

            # Todo Zeta Functions and L functions Section
            '13': 'infinite_product_representation_of_the_zeta_function(z)',
            '14': 'product_of_infinite_product_representation_of_the_zeta_function(z)',

            # Todo Basic library of functions
            '15': 'abs(mpmath.loggamma(z))',
            '16': '1 / (1 + z ^ 2)',
            '17': 'abs(z ** z)',
            '18': 'mpmath.gamma(z)',

            # Todo User Function Utilities
            '19': 'Custom_Function(z)',
            '20': 'product_factory("∏_(n=2)^z [pi*z ∏_(k=2)^z (1 - z^2 / (k^2 * n^2))]")',
            '21': 'product_combination(z)',
            '22': 'product_of_sin_plus_normalized_product_of_sin(z)',
            '23': 'cos_of_product_of_sin(z)',
            '24': 'sin_of_product_of_sin(z)',
            '25': 'cos_of_product_of_product_representation_of_sin(z)',
            '26': 'sin_of_product_of_product_representation_of_sin(z)',

            '27': 'Custom_Riesz_Product_for_Tan(z)',
            '28': 'Dedekind_Eta_Custom(z)',
            '29': 'Custom_Viete_Product_for_Cos(z)',
            '30': 'Half_Base_Viete_Product_for_Sin(z)',
            '31': 'Log_power_base_Viete_Product_for_Sin(z)',
            '32': 'prime_sieve_0_bz_power(z)',
            '33': 'zeros_of_zeta(z)',
            '34': 'Prime_Binary_Output_Waveform_Product(z)',
            '35': 'Riesz_Product_for_Tan_and_Prime_indicator_combination(z)',
            '36': 'Nested_roots_product_for_2(z)',
            '37': 'product_of_product_representation_for_sin_COMPLEX_VARIANT(z)',
        }

        print("Please Select a Function to plot:")
        # Print the menu to the command prompt
        for key, value in Catalog.items():
            print(f'{key}: {value}')

        while True:
            user_input = input('Enter your choice: ')
            if user_input in operations:
                return operations[user_input]
            else:
                print("Invalid choice. Please try again.")

        return operations[f'{user_input}']