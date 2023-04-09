from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from Special_Functions import Special_Functions
from matplotlib.colors import LightSource
from matplotlib.colors import PowerNorm

plt.style.use('dark_background')

class Complex_Plotting :
    """
    A Class for plotting complex functions. This tool provides the plotting functions for 2D and 3D graphics for
    infinite series such as the riemann zeta function, the infinite product of the product representation for
    sin(pi*z/n) as well as any other function that can be formulated in the complex plane.
    """

    def __init__(self):
        """
        Initialization the Special_Functions Object
        """

        #TODO Add variable for 2D vs 3D plot, take the plotting code from Infinite_Prod_of_Prod_Representation_Of_Sin_Complex_Plot.py
        # and incorporate that code into Product_Fractal.py by adding a second create_plot function where it decides
        # which plotting function to use based on the selector

        # TODO Values of n selector
        #self.n = 150
        self.n = 100
        #self.n = 500
        #self.n = 750

        # TODO Values for x & y selector
        self.x_min = 10
        self.x_max = 11
        self.y_max = 0.01
        self.y_min = -0.01

        # self.x_min = 17.5
        # self.x_max = 20.5
        # self.y_max = 0.01
        # self.y_min = -0.01

        # self.x_min = 15
        # self.x_max = 21
        # self.y_max = 0.1366
        # self.y_min = -0.1366

        # self.x_min = 35
        # self.x_max = 49
        # self.y_max = 0.1366
        # self.y_min = -0.1366

        return

    def Colorization(self, color_selection, Z):
        """
        Args:
            color_selection (str): the name of the color map to use
            Z (numpy.ndarray): the array to apply colorization to
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
            colors = np.zeros((self.n, self.n, 3))
            colors[:, :, 0] = np.sin(2 * np.pi * np.real(Z) / 8.0)
            colors[:, :, 1] = np.sin(2 * np.pi * np.real(Z) / 9.0)
            colors[:, :, 2] = np.sin(2 * np.pi * np.real(Z) / 10.0)
            return colors
        elif color_selection == "colors2":
            # Alternate Colorization, it is based upon the angle change around the point of f(z)
            colors = np.zeros((self.n, self.n, 3))
            colors[:, :, 0] = np.sin(2 * np.pi * np.real(Z) / 12)
            colors[:, :, 1] = np.sin(2 * np.pi * np.real(Z) / 14)
            colors[:, :, 2] = np.sin(2 * np.pi * np.real(Z) / 16)
            return colors
        elif color_selection == "colors3":
            # Attempt at mandelbrot level detail with color gradient selector
            colors = np.zeros((self.n, self.n, 3))
            for i in range(self.n):
                for j in range(self.n):
                    colors[i, j, :] = [(i * j) % 256 / 255, (i + j) % 256 / 255, (i * j + i + j) % 256 / 255]
            return colors
        else:
            # apply color selection to the grid map
            cmap = ListedColormap(color_selection)

        # store the colorized grid map and return
        colors = cmap(Z / np.max(Z))

        # TODO Implement On off for lightshading and improve its colorization
        # # Apply shading to the colorized image
        # light = LightSource(azdeg=315, altdeg=10)
        # shaded_colors = light.shade(colors, cmap=plt.cm.hot, vert_exag=1.0,
        #                             norm=PowerNorm(0.3), blend_mode='hsv')

        return colors

    def create_plot(self):
        """
        Args:
            None
        Returns:
            Matlab plot
        """
        #Initialize plot axis and grid point mesh
        X = np.linspace(self.x_min, self.x_max, self.n)
        Y = np.linspace(self.y_min, self.y_max, self.n)
        X, Y = np.meshgrid(X, Y)

        # changed dtype to float
        Z = np.zeros_like(X, dtype=np.float64)

        # calculate special functions object f(z)
        f = special_functions_object.calculate()

        # for loop which plots the point of the selected function f(z)
        for i in range(self.n):
            for j in range(self.n):
                z = complex(X[i, j], Y[i, j])
                Z[i, j] = abs(f(z))

        # TODO Implement nice selector for colorization methods
        shaded_colors = self.Colorization("colors1", Z)
        # colors = self.Colorization("colors2", Z)
        # colors = self.Colorization("colors3", Z)

        # TODO Other coloring algorithms
        # colors = self.Colorization("viridis", Z)
        # colors = self.Colorization("plasma", Z)
        # colors = self.Colorization("magma", Z)

        fig, ax1 = plt.subplots(figsize=(8, 8))

        #TODO Selector for OG colorization vs light source ==============================
        #ax1.imshow(colors, extent=(self.x_min, self.x_max, self.y_min, self.y_max), origin='lower', aspect='auto')
        # ax1.imshow(np.log(Z), extent=(x_min, x_max, y_min, y_max), origin='lower')

        # Plot the shaded image
        ax1.imshow(shaded_colors, extent=(self.x_min, self.x_max, self.y_min, self.y_max), origin='lower',
                   aspect='auto')
        #TODO ===========================================================================

        # Add Title To Plot
        ax1.set_title('Colorization 1')

        # Set tick locators and formatters for the x and y axes
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))  # show tick marks every 2 units
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # format tick labels as integers
        # ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))  # show tick marks every 2 units
        # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))  # format tick labels as integers

        plt.tight_layout()
        plt.show()

        return

if __name__ == "__main__":
    """
    If name main, main function for the Special_Functions Class
    Initializes Special_Functions() object, abd Complex_Plotting() object
    the Special_Functions() object is plotted on the Complex_Plotting() object
    with the create_plot() method
    """
    # instantiate the objects
    special_functions_object = Special_Functions()
    complex_plotting_object = Complex_Plotting()

    # create complex plot from special functions object
    complex_plotting_object.create_plot()
