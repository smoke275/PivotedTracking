# Dubins Path Planner (Pure Python Implementation)

This repository provides a pure Python implementation of the classic Dubins path planner.

## Overview

The **Dubins car** is a simple vehicle model that can only move forward and has a
limited turning radius.  The shortest path between two poses (position and
heading) for such a vehicle can always be decomposed into exactly three
segments.  Each segment is either a left turn (`L`), a straight segment
(`S`) or a right turn (`R`).  Combining these primitives yields six
candidate paths: `LSL`, `LSR`, `RSL`, `RSR`, `RLR` and `LRL`.  The
shortest valid candidate is the Dubins path between two poses.

The original `dubins` package on PyPI wraps a C implementation via Cython.
That code targets old versions of Python and does not compile on Python
3.12+.  This repository re‑implements the core path computation in pure
Python.  It exposes an API compatible with the original `dubins` library
(`shortest_path`, `path`, `path_sample` and the `DubinsPath` class) so
existing code can continue to operate without modification.  Since this
implementation is written in pure Python it works out of the box on
Python 3.12.9 and does not require a compiler or Cython.

## Installation

The package is contained entirely in the `dubins` directory and may be
installed with `pip` once packaged.  For example:

```sh
pip install path/to/dubins_py312
```

Alternatively you can copy the `dubins` folder into your project and
import it directly.

## Basic Usage

```python
import dubins

# Define start and goal configurations (x, y, heading [radians])
q0 = (0.0, 0.0, 0.0)
q1 = (1.0, 1.0, 1.57)  # ~90 degrees

# Vehicle turning radius
rho = 1.0

# Compute the shortest path
path = dubins.shortest_path(q0, q1, rho)

# Sample the path at regular intervals
configurations, distances = path.sample_many(0.1)
```

See the docstrings in `dubins/__init__.py` for full API details.

## License

This library is distributed under the MIT license, the same as the
original `dubins` implementation.