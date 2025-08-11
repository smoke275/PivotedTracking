from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import numpy as np

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "fast_visibility",
        [
            "cpp_extensions/fast_visibility.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            np.get_include(),
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args=[
            '-O3',  # Maximum optimization
            '-march=native',  # Use native CPU instructions
            '-mtune=native',
            '-ffast-math',  # Fast math optimizations
            '-DWITH_THREAD',
        ],
        extra_link_args=['-O3'],
    ),
    # Added polygon exploration accelerated module
    Pybind11Extension(
        "polygon_exploration_cpp",
        [
            "cpp_extensions/polygon_exploration.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args=[
            '-O3',
            '-march=native',
            '-mtune=native',
            '-ffast-math',
        ],
        extra_link_args=['-O3'],
    ),
]

setup(
    name="fast_visibility",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
