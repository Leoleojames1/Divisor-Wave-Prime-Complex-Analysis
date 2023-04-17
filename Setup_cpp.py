

# ------------------------------------------------------------------------------------------------------------
# C++ Plotting Menu
if plot_type == "C++":
    # Set the path to the bindings.cpp file and the directory to place the compiled shared library
    cpp_file = "cplusplus_extensions/bindings.cpp"
    output_dir = "gitprime/cplusplus_extensions"

    # Compile the C++ code
    cmd = f"g++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` {cpp_file} -o {output_dir}/graphics.so"
    subprocess.run(cmd, shell=True, check=True)

    graphics.create_plot_2D()
    # TODO implement plotting from https://github.com/alordash/newton-fractal