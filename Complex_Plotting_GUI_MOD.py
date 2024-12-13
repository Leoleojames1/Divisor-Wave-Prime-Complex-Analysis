"""
Complex_Plotting.py

The Complex_Plotting_GUI_MOD class provides tools for plotting in the complex plane using matplotlib and PyQt5.
It can graph polynomial functions, trigonometric functions, infinite summations, and infinite products 
related to the distribution of prime numbers and the Riemann Hypothesis. The class utilizes methods from 
Special_Functions.py for infinite product formulas.

Date: 04/09/2023
Author: @LeoBorcherding
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, 
    QHBoxLayout, QLabel, QComboBox, QLineEdit, QMessageBox
)

from Special_Functions import Special_Functions  # Ensure this module is in the same directory or properly installed

# Apply a dark background style to all matplotlib plots
plt.style.use('dark_background')


class Complex_Plotting_GUI_MOD:
    """
    A class to create a GUI for plotting complex functions in 2D and 3D using PyQt5 and matplotlib.
    """

    def __init__(self, plot_type='2D', color_map='viridis', normalize_type='linear'):
        """
        Initializes the Complex_Plotting_GUI_MOD class with default plot settings.

        Parameters:
        - plot_type (str): Type of plot ('2D' or '3D').
        - color_map (str): Colormap to use for plotting.
        - normalize_type (str): Type of normalization ('linear', 'log', etc.).
        """
        self.plot_type = plot_type
        self.color_map = color_map
        self.normalize_type = normalize_type

        # Initialize the special functions object
        self.special_functions_object = Special_Functions()

    class MyMplCanvas(FigureCanvas):
        """
        A subclass of FigureCanvas to create a matplotlib figure for embedding in PyQt5.
        """

        def __init__(self, parent=None, width=5, height=4, dpi=100):
            """
            Initializes the matplotlib figure and axes.

            Parameters:
            - parent: Parent widget.
            - width (float): Width of the figure.
            - height (float): Height of the figure.
            - dpi (int): Dots per inch.
            """
            fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = fig.add_subplot(111, projection='3d' if parent.plot_type == "3D" else None)
            super(MyMplCanvas, self).__init__(fig)

        def create_plot_3D(self, special_functions_object, color_map, normalize_type):
            """
            Creates a 3D surface plot of the special function in the complex plane.

            Parameters:
            - special_functions_object: An instance of Special_Functions containing the function to plot.
            - color_map (str): Colormap to use for the surface.
            - normalize_type (str): Normalization type for function values.
            """
            # Clear previous plots
            self.axes.clear()

            # Generate the lambda function array based on normalization
            lambda_function_array = special_functions_object.lambda_function_library(normalize_type)

            # Define grid resolution and limits
            resolution = 0.05  # Adjust for desired resolution
            x_min, x_max = -5, 5
            y_min, y_max = -5, 5

            # Create a grid of complex numbers
            X = np.arange(x_min, x_max, resolution)
            Y = np.arange(y_min, y_max, resolution)
            X, Y = np.meshgrid(X, Y)
            Z = X + 1j * Y

            # Initialize W to store function values
            W = np.zeros_like(X, dtype=np.float64)

            # Compute function values on the grid
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    z = Z[i, j]
                    try:
                        w = float(lambda_function_array(z))
                        if np.isnan(w) or np.isinf(w):
                            W[i, j] = np.nan
                        else:
                            W[i, j] = w
                    except Exception:
                        W[i, j] = np.nan  # Assign NaN for undefined points

            # Plot the surface
            surf = self.axes.plot_surface(X, Y, W, cmap=color_map, linewidth=0, antialiased=False)

            # Set plot labels and title
            self.axes.set_xlabel('Real Axis')
            self.axes.set_ylabel('Imaginary Axis')
            self.axes.set_zlabel('Value')
            self.axes.set_title("3D Plot of Special Function")

            # Add a color bar for reference
            self.figure.colorbar(surf, ax=self.axes, shrink=0.5, aspect=10)

            # Refresh the canvas
            self.draw()

        def create_plot_2D(self, special_functions_object, color_map, normalize_type):
            """
            Creates a 2D heatmap plot of the special function in the complex plane.

            Parameters:
            - special_functions_object: An instance of Special_Functions containing the function to plot.
            - color_map (str): Colormap to use for the heatmap.
            - normalize_type (str): Normalization type for function values.
            """
            # Clear previous plots
            self.axes.clear()

            # Generate the lambda function array based on normalization
            lambda_function_array = special_functions_object.lambda_function_library(normalize_type)

            # Define grid resolution and limits
            resolution = 0.05  # Adjust for desired resolution
            x_min, x_max = -5, 5
            y_min, y_max = -5, 5

            # Create a grid of complex numbers
            X = np.arange(x_min, x_max, resolution)
            Y = np.arange(y_min, y_max, resolution)
            X, Y = np.meshgrid(X, Y)
            Z = X + 1j * Y

            # Initialize W to store function magnitudes
            W = np.zeros_like(X, dtype=np.float64)

            # Compute function magnitudes on the grid
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    z = Z[i, j]
                    try:
                        w = float(lambda_function_array(z))
                        if np.isnan(w) or np.isinf(w):
                            W[i, j] = np.nan
                        else:
                            W[i, j] = w
                    except Exception:
                        W[i, j] = np.nan  # Assign NaN for undefined points

            # Plot the heatmap
            heatmap = self.axes.imshow(
                W, extent=(x_min, x_max, y_min, y_max),
                origin='lower', cmap=color_map, aspect='auto'
            )

            # Set plot labels and title
            self.axes.set_xlabel('Real Axis')
            self.axes.set_ylabel('Imaginary Axis')
            self.axes.set_title("2D Heatmap of Special Function")

            # Add a color bar for reference
            self.figure.colorbar(heatmap, ax=self.axes)

            # Refresh the canvas
            self.draw()

    class MyWindow(QMainWindow):
        """
        The main window class for the Complex Plotting GUI.
        """

        def __init__(self, plot_type='2D', color_map='viridis', normalize_type='linear'):
            """
            Initializes the main window with the plotting canvas and control widgets.

            Parameters:
            - plot_type (str): Type of plot ('2D' or '3D').
            - color_map (str): Colormap to use for plotting.
            - normalize_type (str): Type of normalization ('linear', 'log', etc.).
            """
            super(MyWindow, self).__init__()

            # Initialize plot parameters
            self.plot_type = plot_type
            self.color_map = color_map
            self.normalize_type = normalize_type

            # Initialize the plotting class
            self.plotting_class = Complex_Plotting_GUI_MOD(
                plot_type=self.plot_type,
                color_map=self.color_map,
                normalize_type=self.normalize_type
            )

            # Set up the canvas
            self.canvas = self.plotting_class.MyMplCanvas(self, width=8, height=6, dpi=100)

            # Initialize the special functions object
            self.special_functions = self.plotting_class.special_functions_object

            # Create control widgets
            self.init_ui()

            # Create the initial plot
            self.update_plot()

        def init_ui(self):
            """
            Sets up the UI components: buttons, dropdowns, inputs, etc.
            """
            # Create a button to update the plot
            self.update_button = QPushButton('Update Plot')
            self.update_button.clicked.connect(self.update_plot)

            # Create dropdown for plot type
            self.plot_type_label = QLabel("Plot Type:")
            self.plot_type_dropdown = QComboBox()
            self.plot_type_dropdown.addItems(["2D", "3D"])
            self.plot_type_dropdown.setCurrentText(self.plot_type)

            # Create dropdown for color map
            self.color_map_label = QLabel("Color Map:")
            self.color_map_dropdown = QComboBox()
            self.color_map_dropdown.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'prism'])
            self.color_map_dropdown.setCurrentText(self.color_map)

            # Create dropdown for normalization type
            self.normalize_label = QLabel("Normalization:")
            self.normalize_dropdown = QComboBox()
            self.normalize_dropdown.addItems(['linear', 'log'])
            self.normalize_dropdown.setCurrentText(self.normalize_type)

            # Layout for controls
            control_layout = QHBoxLayout()
            control_layout.addWidget(self.plot_type_label)
            control_layout.addWidget(self.plot_type_dropdown)
            control_layout.addWidget(self.color_map_label)
            control_layout.addWidget(self.color_map_dropdown)
            control_layout.addWidget(self.normalize_label)
            control_layout.addWidget(self.normalize_dropdown)
            control_layout.addWidget(self.update_button)

            # Main layout
            main_layout = QVBoxLayout()
            main_layout.addLayout(control_layout)
            main_layout.addWidget(self.canvas)

            # Set the central widget
            widget = QWidget()
            widget.setLayout(main_layout)
            self.setCentralWidget(widget)

            # Set window properties
            self.setWindowTitle("Complex Plane Plotter")
            self.setGeometry(100, 100, 1000, 800)

        def update_plot(self):
            """
            Updates the plot based on the current settings from the UI.
            """
            try:
                # Retrieve current settings from dropdowns
                self.plot_type = self.plot_type_dropdown.currentText()
                self.color_map = self.color_map_dropdown.currentText()
                self.normalize_type = self.normalize_dropdown.currentText()

                # Update the plotting class parameters
                self.plotting_class.plot_type = self.plot_type
                self.plotting_class.color_map = self.color_map
                self.plotting_class.normalize_type = self.normalize_type

                # Update the plot based on the selected type
                if self.plot_type == "2D":
                    self.canvas.create_plot_2D(
                        self.special_functions,
                        self.color_map,
                        self.normalize_type
                    )
                elif self.plot_type == "3D":
                    self.canvas.create_plot_3D(
                        self.special_functions,
                        self.color_map,
                        self.normalize_type
                    )
                else:
                    QMessageBox.warning(self, "Plot Type Error", f"Unknown plot type: {self.plot_type}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while updating the plot:\n{e}")


def main():
    """
    The main function to run the Complex Plotting GUI application.
    """
    app = QApplication(sys.argv)

    # Default settings (can be modified or extended)
    plot_type = "3D"  # Options: "2D", "3D"
    color_map = "viridis"  # Example colormaps: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'prism'
    normalize_type = "linear"  # Options: 'linear', 'log'

    # Create and show the main window
    window = Complex_Plotting_GUI_MOD.MyWindow(
        plot_type=plot_type,
        color_map=color_map,
        normalize_type=normalize_type
    )
    window.show()

    # Execute the application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
