"""
Complex Plotting
    The Complex Plotting class is a set of tools for plotting in the complex plane utilizing matplotlib.
    The Tool is capable of graphing polynomial functions, trig functions, infinite summations, and infinite
    products in the complex plane. These products are related to the distribution of prime numbers and the Riemann
    Hypothesis. The methods in call methods from the file Special_Functions.py for infinite product formulas.
4/9/2023
@LeoBorcherding
"""
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import ticker
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.colors import PowerNorm

from Special_Functions import Special_Functions

plt.style.use('dark_background')

# -----------------------------------------------------------------------------------------------------------------
class Complex_Plotting :
    """
    A Class for plotting complex functions. This tool provides the plotting functions for 2D and 3D graphics for
    infinite series such as the riemann zeta function, the infinite product of the product representation for
    sin(pi*z/n) as well as any other function that can be formulated in the complex plane.
    """
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, plot_type):
        """
        Initialization the Special_Functions Object
        """
        #TODO Implement Default Values & preselected data ranges for "nice" areas. (pretty areas?)

        #TODO Add variable for 2D vs 3D plot, take the plotting code from Infinite_Prod_of_Prod_Representation_Of_Sin_Complex_Plot.py
        # and incorporate that code into Product_Fractal.py by adding a second create_plot function where it decides
        # which plotting function to use based on the selector

        # TODO Values for x & y selector

        # TODO the following groups are nice areas to explore for the complex plane

        # if user selected 2D plot, graph the plot with the given values
        if plot_type == "2D":

            # Default Values, Copy these if you are going to change them
            self.resolution_2D = 300
            self.x_min_2D = 0
            self.x_max_2D = 14
            self.y_min_2D = -0.8
            self.y_max_2D = 0.8

        # if user selected 2D plot, graph the plot with the given values
        if plot_type == "3D":

            # Default Values, Copy these if you are going to change them
            self.resolution_3D = 0.0899
            self.x_min_3D = 1
            self.x_max_3D = 20
            self.y_min_3D = -3
            self.y_max_3D = 3

        return

    # -----------------------------------------------------------------------------------------------------------------
    def colorization(self, color_selection, Z):
        """
        Args:
            color_selection (str): the name of the color map to use
            Z (numpy.ndarray): the array to apply colorization to
            resolution_2D
        Returns:
        """

        # TODO create better Selector for colorization Methods and Ultimately implement a color slider
        if color_selection == "viridis":
            # Base viridis color map
            cmap = plt.get_cmap('viridis')
        elif color_selection == "plasma":
            # Base plasma color map
            cmap = plt.get_cmap('plasma')
        elif color_selection == "magma":
            # Base magma color map
            cmap = plt.get_cmap('magma')
        elif color_selection == "colors1":
            # Default Colorization, it is based upon the angle change around the point of f(z)
            colors = np.zeros((self.resolution_2D, self.resolution_2D, 3))
            colors[:, :, 0] = np.sin(2 * np.pi * np.real(Z) / 8.0)
            colors[:, :, 1] = np.sin(2 * np.pi * np.real(Z) / 9.0)
            colors[:, :, 2] = np.sin(2 * np.pi * np.real(Z) / 10.0)
            return colors
        elif color_selection == "colors2":
            # Alternate Colorization, it is based upon the angle change around the point of f(z)
            colors = np.zeros((self.resolution_2D, self.resolution_2D, 3))
            colors[:, :, 0] = np.sin(2 * np.pi * np.real(Z) / 12)
            colors[:, :, 1] = np.sin(2 * np.pi * np.real(Z) / 14)
            colors[:, :, 2] = np.sin(2 * np.pi * np.real(Z) / 16)
            return colors
        elif color_selection == "colors3":
            # Attempt at mandelbrot level detail with color gradient selector
            colors = np.zeros((self.resolution_2D, self.resolution_2D, 3))
            for i in range(self.resolution_2D):
                for j in range(self.resolution_2D):
                    colors[i, j, :] = [(i * j) % 256 / 255, (i + j) % 256 / 255, (i * j + i + j) % 256 / 255]
            return colors
        else:
            # apply color selection to the grid map
            cmap = ListedColormap(color_selection)

        # store the colorized grid map and return
        colors = cmap(Z / np.max(Z))
        #colors = cmap((Z - np.min(Z)) / (np.max(Z) - np.min(Z)))

        # TODO Implement On off for lightshading and improve its colorization
        # # Apply shading to the colorized image
        # light = LightSource(azdeg=315, altdeg=10)
        # shaded_colors = \
        #     light.shade(colors, cmap=plt.cm.hot, vert_exag=1.0, norm=PowerNorm(0.3), blend_mode='hsv')
        return colors

    # -----------------------------------------------------------------------------------------------------------------
    def create_plot_2D(self):
        """
        Args:
            resolution_2D:
            x_min_2D:
            x_max_2D:
            y_min_2D:
            y_max_2D:
            color_map:
        Returns:
            Matlab plot
        """
        #Initialize plot axis and grid point mesh
        X = np.linspace(self.x_min_2D, self.x_max_2D, self.resolution_2D)
        Y = np.linspace(self.y_min_2D, self.y_min_2D, self.resolution_2D)
        X, Y = np.meshgrid(X, Y)

        # changed dtype to float
        Z = np.zeros_like(X, dtype=np.float64)

        # calculate special functions object f(z)
        f = special_functions_object.lamda_function_library()

        # for loop which plots the point of the selected function f(z)
        for i in range(self.resolution_2D):
            for j in range(self.resolution_2D):
                z = complex(X[i, j], Y[i, j])
                Z[i, j] = abs(f(z))

        # colors = np.zeros((300, 300, 3))
        # colors[:, :, 0] = np.sin(2 * np.pi * np.real(Z) / 8.0)
        # colors[:, :, 1] = np.sin(2 * np.pi * np.real(Z) / 9.0)
        # colors[:, :, 2] = np.sin(2 * np.pi * np.real(Z) / 10.0)

        # if color_map_2D == "1":
        #colors = self.colorization("colors1", Z)
        # elif color_map_2D == "2":
        #     colors = self.colorization("colors2", Z, resolution_2D)
        # elif color_map_2D == "3":
        #     colors = self.colorization("colors3", Z, resolution_2D)
        # elif color_map_2D == "4":
        #colors = self.colorization("viridis", Z)
        # elif color_map_2D == "5":
        colors = self.colorization("plasma", Z)
        # elif color_map_2D == "6":
        #     colors = self.colorization("magma", Z, resolution_2D)

        fig, ax1 = plt.subplots(figsize=(8, 8))

        # TODO IMPLEMENT 2nd Subplot where 2D fractal is on 1 side, and 3D Fractal is on the other, have this be
        #  optional through user input

        #TODO Selector for OG colorization vs light source ==============================
        ax1.imshow(colors, extent=(self.x_min_2D, self.x_max_2D, self.y_min_2D, self.y_max_2D), interpolation='bicubic', origin='lower', aspect='auto')
        # ax1.imshow(np.log(Z), extent=(x_min_2D, x_max_2D, y_min_2D, y_max_2D), origin='lower')

        # shaded_colors = colors

        # # Plot the shaded image
        # ax1.imshow(shaded_colors, extent=(x_min_2D, x_max_2D, y_min_2D, y_max_2D), origin='lower',
        #            aspect='auto')
        #TODO ===========================================================================

        # Add Title To Plot
        ax1.set_title(f'Colorization {special_functions_object}')

        # Set tick locators and formatters for the x and y axes
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))  # show tick marks every 2 units
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # format tick labels as integers
        # ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))  # show tick marks every 2 units
        # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # format tick labels as integers

        plt.tight_layout()
        plt.show()

        return

    # -----------------------------------------------------------------------------------------------------------------
    def create_plot_3D(self):
        """
        Plotting function.
        The functions for f are commented out for selection.
        Args:
            m - index value used for function when updating and creating plot
        Returns:
            NA
        """

        # calculate special functions object f(z)
        f = special_functions_object.lamda_function_library()

        R = self.resolution_3D
        X = np.arange(self.x_min_3D, self.x_max_3D, R)
        Y = np.arange(self.y_min_3D, self.y_max_3D, R)

        X, Y = np.meshgrid(X, Y)
        xn, yn = X.shape
        W = X * 0

        for xk in range(xn):
            for yk in range(yn):
                try:
                    z = complex(X[xk, yk], Y[xk, yk])
                    w = float(f(z))

                    if w != w:
                        raise ValueError
                    W[xk, yk] = w

                except (ValueError, TypeError, ZeroDivisionError):
                    # can handle special values here
                    pass

        # -----------------------------------------------------------------------------------------------------------------------
        # Set up the plot
        fig = plt.figure(figsize=(12, 10))
        # create 3d subplot
        ax = fig.add_subplot(111, projection='3d')
        # Set initial plot angles
        ax.view_init(elev=30, azim=-45)
        ax.dist = 10
        ax.set_box_aspect((5, 5, 1))
        ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
        # Set the aspect ratio to be 1:1:0.5 # Was originally that ratio
        ax.set_box_aspect((5, 5, 1))

        if W is None:
            # First time creating the plot
            W = W
            #TODO IMPLEMENT COLOR MAP SELECTION
            #TODO IMPLEMENT MANDELBROT 3D Fractal Mapping
            ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet)
            # ax.plot_wireframe(X, Y, W, rstride=5, cstride=5)

        else:
            # Update the existing plot
            W = W
            ax.clear()
            ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet)
            # ax.plot_wireframe(X, Y, W, rstride=5, cstride=5)

        ax.set_xlabel('Real Axis')
        ax.set_ylabel('Imaginary Axis')
        ax.set_zlabel('Value')
        ax.set_title(f"Product of Sin(x) via Sin(x) product Representation")
        plt.draw()
        plt.show()

        return

# -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    If name main, main function for the SpecialFunctions and ComplexPlotting Classes
    Initializes SpecialFunctions() object, and ComplexPlotting() object
    the SpecialFunctions() object is plotted on the ComplexPlotting() object
    with the create_plot_2D() or create_plot_3D() method
    """

    # get user input for plot type
    print("Enter 2D for 2D plot or 3D for 3D plot: ")
    plot_type = input()

    # instantiate the objects
    special_functions_object = Special_Functions(plot_type)
    complex_plotting_object = Complex_Plotting(plot_type)

    # if user selected 2D plot, graph the plot with the given values
    if plot_type == "2D":
        complex_plotting_object.create_plot_2D()

    # if user selected 2D plot, graph the plot with the given values
    if plot_type == "3D":
        complex_plotting_object.create_plot_3D()

    # # 2D Complex Plot User Dimension Selections
    # if plot_type == "2D":
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested Values
    #     resolution_2D_dict = \
    #         {
    #          "1": 100,
    #          "2": 150,
    #          "3": 200,
    #          "4": 250,
    #          "5": 330,
    #          "6": 500,
    #          "7": 750,
    #         }
    #
    #     print("Here are some suggested resolutions, please enter any value "
    #           "from 1 to infinity, larger numbers will cause longer run time:")
    #     # print suggested values from dictionary
    #     for key in resolution_2D_dict:
    #         print(f'"{key}" : "{resolution_2D_dict[key]}"')
    #
    #     # Get resolution value from user input
    #     resolution_2D = int(input('Select resolution_2D: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested x_min_2D Values
    #     x_min_2D_dict = \
    #         {
    #          "1": 0,
    #          "2": 2,
    #          "3": 4,
    #          "4": 10,
    #          "5": 16,
    #          "6": 32,
    #          "7": 108,
    #         }
    #
    #     print("Here are some suggested x_min_2D values, please enter any value "
    #           "from 1 to infinity, larger numbers will cause longer run time:")
    #     for key in x_min_2D_dict:
    #         print(f'"{key}" : "{x_min_2D_dict[key]}"')
    #     # Get x-axis min range from user input
    #     x_min_2D = float(input('Select x_min_2D axis range value: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested x_max_2D Values
    #     x_max_2D_dict = \
    #         {
    #          "1": 5,
    #          "2": 8,
    #          "3": 14,
    #          "4": 18,
    #          "5": 28,
    #          "6": 36,
    #          "7": 48,
    #          "8": 57,
    #          "9": 68,
    #          "10": 79,
    #          "11": 84,
    #         }
    #     print("Here are some suggested x_max_2D values, please enter any value "
    #           "from 1 to infinity, larger numbers will cause longer run time:")
    #     for key in x_max_2D_dict:
    #         print(f'"{key}" : "{x_max_2D_dict[key]}"')
    #     # Get x-axis min range from user input
    #     x_max_2D = float(input('Select x_max_2D axis range value: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested y_min_2D Values
    #     y_min_2D_dict = \
    #         {
    #          "1": -0.5,
    #          "2": -0.1366,
    #          "3": -0.7766,
    #          "4": -0.01,
    #         }
    #     print("Here are some suggested y_min_2D values, please enter any value "
    #           "from 1 to infinity, larger numbers will cause longer run time:")
    #     for key in y_min_2D_dict:
    #         print(f'"{key}" : "{y_min_2D_dict[key]}"')
    #     # Get x-axis min range from user input
    #     y_min_2D = float(input('Select y_min_2D axis range value: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested y_max_2D Values
    #     y_max_2D_dict = \
    #         {
    #          "1": 0.5,
    #          "2": 0.1366,
    #          "3": 0.7766,
    #          "4": 0.01,
    #         }
    #     print("Here are some suggested y_max_2D values, please enter any value")
    #     print("from 1 to infinity, larger numbers will cause longer run time")
    #
    #     for key in y_max_2D_dict:
    #         print(f'"{key}" : "{y_max_2D_dict[key]}"')
    #     # Get x-axis min range from user input
    #     y_max_2D = float(input('Select y_max_2D axis range value: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested Color Map Values
    #     color_map_dict = \
    #         {
    #          "1": "colors1",
    #          "2": "colors2_exp",
    #          "3": "colors3_exp",
    #          "4": "viridis",
    #          "5": "plasma",
    #          "6": "magma",
    #         }
    #     print("Here are the color map functions, please enter the number on the left")
    #     print("to choose that color for the render.")
    #
    #     for key in color_map_dict:
    #         print(f'"{key}" : "{color_map_dict[key]}"')
    #     # Get x-axis min range from user input
    #     color_map_2D = input('Select color map (default or custom): \n')
    #
    # # if user selected 2D plot, graph the plot with the given values
    # if plot_type == "2D":
    #     complex_plotting_object.create_plot_2D(
    #         resolution_2D=resolution_2D, x_min_2D=x_min_2D,
    #         x_max_2D=x_max_2D, y_min_2D=y_min_2D, y_max_2D=y_max_2D, color_map_2D=color_map_2D
    #     )


        # y_min_3D = -4
        # y_max_3D = 4

    # # 3D Complex Plot User Dimension Selections
    # if plot_type == "3D":
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested Values
    #     resolution_3D_dict = \
    #         {
    #          "1": 0.0901,
    #          "2": 0.175,
    #         }
    #
    #     print("Here are some suggested resolutions, please enter any value "
    #           "from 1 to infinity, larger numbers will cause longer run time:")
    #     # print suggested values from dictionary
    #     for key in resolution_3D_dict:
    #         print(f'"{key}" : "{resolution_3D_dict[key]}"')
    #
    #     # Get resolution value from user input
    #     resolution_3D = input('Select resolution_3D: \n')
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested x_min_3D Values
    #     x_min_3D_dict = \
    #         {
    #          "1": 0,
    #          "2": 1,
    #          "3": 2,
    #         }
    #
    #     print("Here are some suggested x_min_3D values, please enter any value "
    #           "from 1 to infinity, larger numbers will cause longer run time:")
    #     for key in x_min_3D_dict:
    #         print(f'"{key}" : "{x_min_3D_dict[key]}"')
    #     # Get x-axis min range from user input
    #     x_min_3D = int(input('Select x_min_3D axis range value: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested x_max_3D Values
    #     x_max_3D_dict = \
    #         {
    #          "1": 5,
    #          "2": 8,
    #          "3": 14,
    #          "4": 18,
    #          "5": 28,
    #          "6": 36,
    #          "7": 48,
    #          "8": 57,
    #          "9": 68,
    #          "10": 79,
    #          "11": 84,
    #         }
    #     print("Here are some suggested x_max_3D values, please enter any value "
    #           "from 1 to infinity, larger numbers will cause longer run time:")
    #     for key in x_max_3D_dict:
    #         print(f'"{key}" : "{x_max_3D_dict[key]}"')
    #     # Get x-axis min range from user input
    #     x_max_3D = int(input('Select x_max_3D axis range value: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested y_min_3D Values
    #     y_min_3D_dict = \
    #         {
    #          "1": -8,
    #          "2": -4,
    #          "3": -2,
    #         }
    #     print("Here are some suggested y_min_3D values, please enter any value "
    #           "from 1 to infinity, larger numbers will cause longer run time:")
    #     for key in y_min_3D_dict:
    #         print(f'"{key}" : "{y_min_3D_dict[key]}"')
    #     # Get x-axis min range from user input
    #     y_min_3D = int(input('Select y_min_3D axis range value: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested y_max_3D Values
    #     y_max_3D_dict = \
    #         {
    #          "1": 2,
    #          "2": 4,
    #          "3": 8,
    #         }
    #     print("Here are some suggested y_max_3D values, please enter any value")
    #     print("from 1 to infinity, larger numbers will cause longer run time")
    #
    #     for key in y_max_3D_dict:
    #         print(f'"{key}" : "{y_max_3D_dict[key]}"')
    #     # Get x-axis min range from user input
    #     y_max_3D = int(input('Select y_max_3D axis range value: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested m coefficient Values
    #     m_dict = \
    #         {
    #          "1": 0.086,
    #          "2": 0.136,
    #          "3": 0.246,
    #          "4": 0.356,
    #          "5": 0.467,
    #          "6": 0.578,
    #          "7": 0.688,
    #          "8": 0.788,
    #          "9": 0.824,
    #          "10": 0.964,
    #          "11": 1,
    #         }
    #     print("Here are the m magnification coefficients suggestion, "
    #           "please enter a value for m, negative values are accepted: ")
    #
    #     for key in m_dict:
    #         print(f'"{key}" : "{m_dict[key]}"')
    #     # Get x-axis min range from user input
    #     m_3D = float(input('Select m magnification coefficient: \n'))
    #
    #     # ------------------------------------------------------------------------------------------------------------
    #     # Print Suggested m coefficient Values
    #     beta_dict = \
    #         {
    #          "1": 0.086,
    #          "2": 0.246,
    #          "3": 0.467,
    #          "4": 0.688,
    #          "5": 0.824,
    #          "6": 1,
    #          "7": 1.086,
    #          "8": 1.246,
    #          "9": 1.467,
    #          "10": 1.688,
    #          "11": 1.824,
    #          "12": 2,
    #         }
    #     print("Here are the beta magnification coefficients suggestion, "
    #           "please enter a value for m, negative values are accepted: ")
    #
    #     for key in beta_dict:
    #         print(f'"{key}" : "{beta_dict[key]}"')
    #     # Get x-axis min range from user input
    #     beta_3D = float(input('Select beta magnification coefficient: \n'))
