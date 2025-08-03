"""Setuptools configuration for the pure Python Dubins path planner."""

from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", encoding="utf-8") as fh:
    long_desc = fh.read()

setup(
    name="dubins",
    version="1.0.2",
    description="Pure Python Dubins path planner compatible with the original dubins API",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="Andrew Walker and contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    include_package_data=True,
)