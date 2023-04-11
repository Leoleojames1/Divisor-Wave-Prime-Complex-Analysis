import math
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab
import numpy as np
import mpmath
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

mpmath.dps = 5
plt.style.use('dark_background')

#-----------------------------------------------------------------------------------------------------------------------
# Normalized Product of Sin(x) via Sin(x) product Representation, normalized through fraction on gamma function.
def normalized_double_product_for_product_of_sin_representation(z, beta, m):
    """
    ∏_(n=2)^x beta*pi*n*(∏_(k=2)^x(1-x^2/(k^2*n^2)))/(∏_(n=2)^x(pi*n*(∏_(k=2)^x(1-x^2/(k^2*n^2))))!
    """
    z_real = np.real(z)
    z_imag = np.imag(z)
    numerator = abs(np.prod([beta * (z_real / n) * (
                (z_real * math.pi + 1j * z_imag * math.pi) * np.prod([1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
                             range(2, int(z_real) + 1)]))
    denominator = abs(np.prod([beta * (z_real / n) * (
                (z_real * math.pi + 1j * z_imag * math.pi) * np.prod([1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
                               range(2, int(z_real) + 1)]))
    normalized_fraction = (numerator ** (-m) / scipy.special.gamma(denominator ** (-m)))
    if (z_real % 1 == 0) and (z_imag % 1 == 0):
        print(z, ": ", normalized_fraction)
    return normalized_fraction

#-----------------------------------------------------------------------------------------------------------------------
# Product of Sin(x) via Sin(x) product Representation.
def product_of_product_for_product_of_sin_representation(z, beta, m):
    """
    ∏_(n=2)^x { abs ( pi*x [ ∏_(k=1)^x ( 1 - x^2 / (k^2 * n^2) ) ] ) }
    """
    z_real = np.real(z)
    z_imag = np.imag(z)
    result = abs(np.prod([beta * (z_real / n) * (
                (z_real * math.pi + 1j * z_imag * math.pi) * np.prod([1 - ((z_real + 1j * z_imag) ** 2) / (n ** 2 * k ** 2) for k in range(2, int(z_real) + 1)])) for n in
                             range(2, int(z_real) + 1)])) ** (-m)
    if (z % 1 == 0):
        print(z, ": ", result)
    return result

#-----------------------------------------------------------------------------------------------------------------------
# Define the function that creates the plot
def create_plot(m):
    """
    Plotting function.
    The functions for f are commented out for selection.
    Args:
        m - index value used for function when updating and creating plot
    Returns:
        NA
    """
    #f = lambda z: product_of_product_for_product_of_sin_representation(z, 0.077, m=0.136)
    f = lambda z: normalized_double_product_for_product_of_sin_representation(z, 0.077, m=0.136)

    #f = lambda z: abs(mpmath.loggamma(z))
    #f = lambda z: 1/(1+z^2)
    #f = lambda z: abs(z**z)
    #f = mpmath.gamma(z)

    X = np.arange(1, 16, 0.1)
    Y = np.arange(-4, 4, 0.1)

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

    if W is None:
        # First time creating the plot
        W = W
        ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet)
        #ax.plot_wireframe(X, Y, W, rstride=5, cstride=5)

    else:
        # Update the existing plot
        W = W
        ax.clear()
        ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.jet)
        #ax.plot_wireframe(X, Y, W, rstride=5, cstride=5)

    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.set_zlabel('Value')
    ax.set_title(f"Product of Sin(x) via Sin(x) product Representation with m={m:.2f}")
    plt.draw()

#-----------------------------------------------------------------------------------------------------------------------
# Set up the plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111,projection='3d')
# Set the aspect ratio to be 1:1:0.5 # Was originally that ratio
ax.set_box_aspect((5, 5, 1))

create_plot(1)
m_slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
m_slider = Slider(m_slider_ax, 'm', valmin=0, valmax=2, valinit=0.136, valstep=0.01)

#-----------------------------------------------------------------------------------------------------------------------
def update(val):
    create_plot(m_slider.val)

m_slider.on_changed(update)
plt.show()