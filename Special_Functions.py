"""
Special Functions
    The Special functions class is a set of special functions in the complex plane involving infinite products in the
    complex plane. These products are related to the distribution of prime numbers and the Riemann Hypothesis. The
    methods in this class are called by the file Complex_Plotting.py for complex the creation of complex plots.
4/9/2023
@LeoBorcherding
"""

import cmath
import math
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
    def __init__(self):
        """
        Initialization the Special_Functions Object, defines self arg m = desired magnification exponent
        """

        # TODO Nice Values of m include m= 0.096, 0.136, 0.187, 0.264, 0.378, 0.487, 0.596, 0.775, 0.844, 0.956
        # self.m = 1
        self.m = 0.096

        # Regex for identifying product symbols TODO Implement Summation Factory Function
        self.pattern = r'([a-z]+)_\(([a-z]+)=([0-9]+)\)\^\(([a-z]+)\=([0-9]+)\)\s*\[(.*)\]'

        return

    # -----------------------------------------------------------------------------------------------------------------
    def product_of_product_representation_for_sin(self, z, m):
        """
        Computes the product of the product representation for sin(z).
        ∏_(n=2)^x (pi*x) ∏_(n=2)^x (1-(x^2)/(i^2)(n^2))
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (float): The product of the product representation for sin(z).
        """
        z_real = np.real(z)
        z_imag = np.imag(z)
        result = abs(np.prod(
            [(z_imag * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def natural_logarithm_of_infinite_product_of_product_representation_for_sin(self,z, m):
        """
        Computes the natural logarithm of the normalized infinite product of product representation of sin(z).
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (complex): The natural logarithm of the normalized infinite product of product representation of sin(z).
        """
        z_real = np.real(z)
        z_imag = np.imag(z)
        result = abs(np.prod(
            [(z_imag * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-m)
        # TODO FUNCTION TRANSFORMATION SELECTOR
        log_result = cmath.log(result)
        #log_result=cmath.log(result) # scipy.special.gamma(
        #return result / np.log(z)
        return log_result

    # -----------------------------------------------------------------------------------------------------------------
    def gamma_of_infinite_product_of_product_representation_for_sin(self, z, m):
        """
        Args:
        Returns:
        """
        z_real = np.real(z)
        z_imag = np.imag(z)
        result = abs(np.prod(
            [(z_imag * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        gamma_result = scipy.special.gamma(result)
        #log_result=cmath.log(result) # scipy.special.gamma(
        return gamma_result

    # -----------------------------------------------------------------------------------------------------------------
    def normalized_infinite_product_of_product_representation_for_sin(self, z, m):
        """
        Args:
        Returns:
        """
        z_real = np.real(z)
        z_imag = np.imag(z)
        result = abs(np.prod(
            [(z_imag * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        fraction_result = result / math.factorial(math.ceil(result))
        #log_result = scipy.special.gamma(fraction_result)
        #log_result=cmath.log(result) # scipy.special.gamma(
        return fraction_result

    # -----------------------------------------------------------------------------------------------------------------
    def factored_form_for_product_of_product_representation_for_sin(self, z, m):
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
            [(z_imag * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [(k*n - (z_real + 1j * z_imag))*(k*n + (z_real + 1j * z_imag)) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def factored_form_for_product_of_product_representation_for_sin(self, z, m):
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
            [(z_imag * z_real / n) * ((z_real * math.pi + 1j * z_imag * math.pi) * np.prod(
                [(k*n - (z_real + 1j * z_imag))*(k*n + (z_real + 1j * z_imag)) / (n ** 2 * k ** 2)
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def infinite_product_representation_of_the_zeta_function(self, z, m):
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
            abs((z_imag * z_real)*(2**(z_real + 1j * z_imag) * math.pi**((z_real + 1j * z_imag) - 1)
                * np.sin(math.pi * (z_real + 1j * z_imag) / 2) * scipy.special.gamma(1 - (z_real + 1j * z_imag))
                    * np.prod([(2**(1-(z_real + 1j * z_imag))) / (k**(1 - (z_real + 1j * z_imag)))
                        for k in range(1, int(z_real) + 1)]))) ** (-m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def product_of_infinite_product_representation_of_the_zeta_function(self, z, m):
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
                            for k in range(1, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def gamma_form_product_of_product_representation_for_sin(self, z, m):
        """
        Computes the product of the product representation for sin(z).
         z / [Γ(z) * Γ(1-z)] * e^(-γz^2) * ∏_(n=2)^(∞) e^(z^2 / n^2)
        Args:
            z (complex): A complex number to evaluate.
            m (float): A constant value for the result.
        Returns:
            (float): The product of the product representation for sin(z).
        """
        mascheroni = 0.57721566490153286060651209008240243104215933593992359880
        z_real = np.real(z)
        z_imag = np.imag(z)
        result = abs(np.prod(
            [(z_imag * z_real / n) * scipy.special.gamma(z_real + 1j * z_imag)
                * scipy.special.gamma(1 - (z_real + 1j * z_imag))
                    * cmath.exp((-((z_real + 1j * z_imag) ** 2))) * np.prod(
                        [(cmath.exp(((z_real + 1j * z_imag)**2/(n**2)*(k**2))))
                            for k in range(2, int(z_real) + 1)]) for n in range(2, int(z_real) + 1)])) ** (-m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def Custom_Function(self, z, m):
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
                 for k in range(2, int(z_real) + 1)])) for n in range(2, int(z_real) + 1)])) ** (-m)
        #TODO FUNCTION TRANSFORMATION SELECTOR
        # right now there are multiple methods for what is a graphing technique of transforming the product with logs and factorials.
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def calculate(self):
        """
        Args:
            None, will eventually be user_input
        Colorization:
        """
        # TODO implement real user input
        # for now users should change this variable to select the function they would like to plot.
        # the function product_of_product_representation_for_sin and Custom_Function are identical however
        # Custom_Function is multiplied by an imaginary scalar for magnification
        # TODO fix the product_factory function, currently broken but will allow users to design their own functions

        user_input = '1'

        operations = {
            '1': lambda z: self.product_of_product_representation_for_sin(z, self.m),
            '2': lambda z: self.natural_logarithm_of_infinite_product_of_product_representation_for_sin(z, self.m),
            '3': lambda z: self.gamma_of_infinite_product_of_product_representation_for_sin(z, self.m),
            '4': lambda z: self.normalized_infinite_product_of_product_representation_for_sin(z, self.m),
            '5': lambda z: self.factored_form_for_product_of_product_representation_for_sin(z, self.m),
            '6': lambda z: self.infinite_product_representation_of_the_zeta_function(z, self.m),
            '7': lambda z: self.product_of_infinite_product_representation_of_the_zeta_function(z, self.m),
            '8': lambda z: self.gamma_form_product_of_product_representation_for_sin(z, self.m),
            '9': lambda z: self.Custom_Function(z, self.m),
            '10': lambda z: self.product_factory("∏_(n=2)^z [pi*z ∏_(k=2)^z (1 - z^2 / (k^2 * n^2))]"),
        }
        return operations[f'{user_input}']

    # -----------------------------------------------------------------------------------------------------------------
    def product_factory(self, product_string_arg):
        """
        Args:
            None.
        Returns:
             None.
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