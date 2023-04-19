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

        normalized = True
        self.im_mag = False

        # if user selected 2D plot, graph the plot with the given values
        if plot_type == "2D":
            # Default Values, Copy these if you are going to change them
            #self.m = 0.096
            #Default for function 3
            self.m = 0.0708
            if normalized == True:
                self.beta = 0.077
            else:
                self.beta = 4.577

        # if user selected 3D plot, graph the plot with the given values
        if plot_type == "3D":
            # Default Values, Copy these if you are going to change them
            # self.m = 0.158
            #Default for function 3
            self.m = 0.118
            # self.m = 0.0708
            # self.m = 0.0008
            if normalized == True:
                #default for function 3
                self.beta = 0.077
            else:
                self.beta = 4.577
                #Good for Sin
                #self.beta = 0.677
                #self.beta = 4.577

        # if user selected 3D plot, graph the plot with the given values
        if plot_type == "PyQt":
            # Default Values, Copy these if you are going to change them
            self.m = 0.158
            if normalized == True:
                self.beta = 0.077
            else:
                self.beta = 4.577
                #Good for Sin
                #self.beta = 0.677
                #self.beta = 4.577

        # Set beta to 1 for no magnification
        #self.beta = 1

        # Regex for identifying product symbols TODO Implement Summation Factory Function
        self.pattern = r'([a-z]+)_\(([a-z]+)=([0-9]+)\)\^\(([a-z]+)\=([0-9]+)\)\s*\[(.*)\]'

        return

    # -----------------------------------------------------------------------------------------------------------------------
    # Product of Sin(x) via Sin(x) product Representation.
    def product_of_sin(self, z):
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

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        # calculate infinite product
        result = abs(np.prod(
            [self.beta * (z_real * imaginary_magnification / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        return result

    # -----------------------------------------------------------------------------------------------------------------------
    # Normalized Product of Sin(x) via Sin(x) product Representation, normalized through fraction on gamma function.
    def normalized_product_of_sin(self, z):
        """
        Computes the normalized infinite product of sin(pi*z/n)
        f(x) = ∏_(n=2)^x { abs ( sin(pi*x/n) ) ] ) } / ∏_(n=2)^x { abs ( sin(pi*x/n) ) ] ) }!
        Args:
            z (complex): A complex number to evaluate.
        Returns:
            (float): The normalized product of sin(pi*z/n).
        """
        # initialize complex components
        z_real = np.real(z)
        z_imag = np.imag(z)

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        # calculate numerator of normalized fraction
        numerator = abs(np.prod(
            [self.beta * (z_real * imaginary_magnification / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        # calculate numerator of normalized fraction
        denominator = abs(np.prod(
            [self.beta * (z_real / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                for n in range(2, int(z_real) + 1)])) ** (-self.m)

        # calculate normalized fraction
        normalized_fraction = (numerator ** (-self.m) / scipy.special.gamma(denominator ** (-self.m)))

        return normalized_fraction

    # -----------------------------------------------------------------------------------------------------------------
    def product_of_product_representation_for_sin(self, z):
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

        # check for imaginary magnification
        if self.im_mag is True:
            imaginary_magnification = z_imag
        else:
            imaginary_magnification = 1

        power_integer = 2

        # calculate the double infinite product via the double for loop
        result = abs(np.prod(
            [self.beta * (imaginary_magnification * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z) ** power_integer) / (n ** power_integer * k ** power_integer)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)

        # # print function values for bug testing
        # if (z_real % 1 == 0) and (z_imag % 1 == 0):
        #     print(z, ": ", result)

        return result

    # -----------------------------------------------------------------------------------------------------------------------
    # Normalized Product of Sin(x) via Sin(x) product Representation, normalized through fraction on gamma function.
    def normalized_product_of_product_representation_for_sin(self, z):
        """
        f(x) = ∏_(n=2)^x beta*pi*n*(∏_(k=2)^x(1-x^2/(k^2*n^2))) / (∏_(n=2)^x(pi*n*(∏_(k=2)^x(1-x^2/(k^2*n^2)))) !
        """
        z_real = np.real(z)
        z_imag = np.imag(z)

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

        # # TODO print function values for bug testing
        # if (z_real % 1 == 0) and (z_imag % 1 == 0):
        #     print(z, ": ", normalized_fraction)

        return normalized_fraction

    # -----------------------------------------------------------------------------------------------------------------
    def natural_logarithm_of_product_of_product_representation_for_sin(self, z):
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
    def gamma_of_product_of_product_representation_for_sin(self, z):
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
    def gamma_form_product_of_product_representation_for_sin(self, z):
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
    def infinite_product_representation_of_the_zeta_function(self, z):
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
    def product_of_infinite_product_representation_of_the_zeta_function(self, z):
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
    def product_combination(self, z):
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

        z_real = np.real(z)
        z_imag = np.imag(z)

        # normalized double infinite product for loop
        combination = np.cos(abs(np.prod([self.beta * (z_real / n) * (
                (z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
            [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
                                 range(2, int(z_real) + 1)])))
        # denominator = abs(np.prod([self.beta * (z_real / n) * (
        #         (z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
        #     [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
        #                            range(2, int(z_real) + 1)]))
        # normalized_fraction = (numerator ** (-self.m) / scipy.special.gamma(denominator ** (-self.m)))

        # combination = np.cos(normalized_fraction)

        return combination

    # -----------------------------------------------------------------------------------------------------------------
    def product_of_sin_plus_normalized_product_of_sin(self, z):
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
    def cos_of_product_of_sin(self, z):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        # normalized double infinite product for loop
        result = np.cos(abs(np.prod([self.beta * (z_real / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                              for n in range(2, int(z_real) + 1)])) ** (-self.m))

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def sin_of_product_of_sin(self, z):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        # normalized double infinite product for loop
        result = np.sin(abs(np.prod([self.beta * (z_real / n) * np.sin(math.pi * (z_real + 1j * z_imag) / n)
                              for n in range(2, int(z_real) + 1)])) ** (-self.m))

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def cos_of_product_of_product_representation_of_sin(self, z):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        # normalized double infinite product for loop
        result = np.cos(abs(np.prod([self.beta * (z_real / n) * (
                (z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
            [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
                                 range(2, int(z_real) + 1)])))

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def sin_of_product_of_product_representation_of_sin(self, z):
        """ A method to multiply two infinite products.
        Args:
            z: Complex z which isn't converted to z_real & z_imag in this method
            beta: magnification value as the lead coefficient
            m: exponential magnification coefficient
        """

        z_real = np.real(z)
        z_imag = np.imag(z)

        # normalized double infinite product for loop
        result = np.sin(abs(np.prod([self.beta * (z_real / n) * (
                (z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
            [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
                                 range(2, int(z_real) + 1)])))

        return result

    # -----------------------------------------------------------------------------------------------------------------
    def Custom_Function(self, z):
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
            [((1j * z_imag * z_real) / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z_real + 1j * z_imag)**(2)) / (n ** (2) * k ** (2))
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-self.m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

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
    def lamda_function_library(self):
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
            '1': lambda z: self.product_of_sin(z),
            '2': lambda z: self.normalized_product_of_sin(z),

            '3': lambda z: self.product_of_product_representation_for_sin(z),
            '4': lambda z: self.normalized_product_of_product_representation_for_sin(z),

            '5': lambda z: self.natural_logarithm_of_product_of_product_representation_for_sin(z),
            '6': lambda z: self.gamma_of_product_of_product_representation_for_sin(z),
            '7': lambda z: self.factored_form_for_product_of_product_representation_for_sin(z),
            '8': lambda z: self.gamma_form_product_of_product_representation_for_sin(z),

            # Todo Zeta Functions and L functions Section
            '9': lambda z: self.infinite_product_representation_of_the_zeta_function(z),
            '10': lambda z: self.product_of_infinite_product_representation_of_the_zeta_function(z),

            # Todo Basic library of functions
            '11': lambda z: abs(mpmath.loggamma(z)),
            '12': lambda z: 1 / (1 + z ^ 2),
            '13': lambda z: abs(z ** z),
            '14': lambda z: mpmath.gamma(z),

            # Todo User Function Utilities
            '15': lambda z: self.Custom_Function(z),
            '16': lambda z: self.product_factory("∏_(n=2)^z [pi*z ∏_(k=2)^z (1 - z^2 / (k^2 * n^2))]"),
            '17': lambda z: self.product_combination(z),
            '18': lambda z: self.product_of_sin_plus_normalized_product_of_sin(z),
            '19': lambda z: self.cos_of_product_of_sin(z),
            '20': lambda z: self.sin_of_product_of_sin(z),
            '21': lambda z: self.cos_of_product_of_product_representation_of_sin(z),
            '22': lambda z: self.sin_of_product_of_product_representation_of_sin(z),

        }

        Catalog = {
            # Todo Special Infinite Products section
            '1': 'product_of_sin',
            '2': 'normalized_product_of_sin',

            '3': 'product_of_product_representation_for_sin',
            '4': 'normalized_product_of_product_representation_for_sin',

            '5': 'natural_logarithm_of_infinite_product_of_product_representation_for_sin',
            '6': 'gamma_of_infinite_product_of_product_representation_for_sin',

            '7': 'factored_form_for_product_of_product_representation_for_sin',

            '8': 'gamma_form_product_of_product_representation_for_sin',

            # Todo Zeta Functions and L functions Section
            '9': 'infinite_product_representation_of_the_zeta_function',
            '10': 'product_of_infinite_product_representation_of_the_zeta_function',

            # Todo Basic library of functions
            '11': 'abs(mpmath.loggamma(z))',
            '12': '1/(1+z^2)',
            '13': 'abs(z**z)',
            '14': 'mpmath.gamma(z)',

            # Todo User Function Utilities
            '15': 'Custom_Function',
            '16': 'product_factory',
            '17': 'product_combination',
            '18': 'product_of_sin_plus_normalized_product_of_sin',
            '19': 'cos_of_product_of_sin',
            '20': 'sin_of_product_of_sin',
            '21': 'cos_of_product_of_sin_plus_normalized_product_of_sin',
            '22': 'sin_of_product_of_sin_plus_normalized_product_of_sin',

        }

        print("Please Select a Function to plot:")
        # Print the menu to the command prompt
        for key in Catalog:
            print(f'{key}: {Catalog[key]}')

        user_input = input('Enter your choice: ')

        return operations[f'{user_input}']