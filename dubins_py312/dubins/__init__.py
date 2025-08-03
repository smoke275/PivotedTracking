"""Pure Python Dubins path planner compatible with the original `dubins` API.

This module provides a forward‑only Dubins car path planner implemented in
pure Python.  It reimplements the functionality of the original
``dubins`` package, which wraps a C library using Cython and therefore
does not compile on modern Python releases (e.g. Python 3.12).  The
public API mirrors that of the original package: the top‑level
``shortest_path``, ``path`` and ``path_sample`` functions as well as the
``DubinsPath`` class.  See the docstrings below for usage examples.

The Dubins car model describes a vehicle that can move forward and turn
left or right with a bounded minimum turning radius, but cannot move
backwards.  The shortest path between two configurations (position and
heading) always consists of three segments drawn from ``L`` (left turn),
``S`` (straight line) and ``R`` (right turn).  The six candidate
combinations are ``LSL``, ``LSR``, ``RSL``, ``RSR``, ``RLR`` and ``LRL``.
The planner evaluates each candidate and selects the one with the
smallest length.

This implementation is self contained and does not depend on NumPy.
Angles are expressed in radians and wrapped into the range ``[0, 2π)``.
Coordinates use a right‑handed 2D coordinate system where heading
``0`` points along the positive X axis and positive angles are measured
counter‑clockwise.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Tuple

# Path type enumerations matching the original C implementation
LSL = 0
LSR = 1
RSL = 2
RSR = 3
RLR = 4
LRL = 5

# Segment type sequences for each path type (L/R/S)
_SEGMENT_SEQUENCES = {
    LSL: ('L', 'S', 'L'),
    LSR: ('L', 'S', 'R'),
    RSL: ('R', 'S', 'L'),
    RSR: ('R', 'S', 'R'),
    RLR: ('R', 'L', 'R'),
    LRL: ('L', 'R', 'L'),
}

def mod2pi(theta: float) -> float:
    """Return ``theta`` wrapped into ``[0, 2π)``.

    Parameters
    ----------
    theta : float
        Angle in radians.

    Returns
    -------
    float
        Equivalent angle in ``[0, 2π)``.
    """
    twopi = 2.0 * math.pi
    # Use fmod and add offset to handle negative values correctly
    r = theta % twopi
    return r + twopi if r < 0 else r


class DubinsPath:
    """Representation of a Dubins path between two configurations.

    A ``DubinsPath`` instance stores the normalised segment parameters,
    the path type and the start configuration.  It exposes methods to
    query path properties and to sample configurations along the path.

    Instances of this class should not be constructed directly; instead
    use :func:`shortest_path` or :func:`path` to compute a path between
    two configurations.

    Parameters
    ----------
    qi : tuple of float
        The initial configuration ``(x, y, heading)``.
    path_type : int
        An integer in ``0..5`` indicating the candidate path type
        (`LSL`, `LSR`, `RSL`, `RSR`, `RLR`, `LRL`).
    params : iterable of float
        A length‑3 sequence containing the normalised lengths ``t``,
        ``p`` and ``q`` of the three segments.  The actual segment
        lengths in world units are obtained by multiplying each element
        by ``rho``.
    rho : float
        The minimum turning radius of the vehicle.
    """

    def __init__(self, qi: Tuple[float, float, float], path_type: int,
                 params: Iterable[float], rho: float) -> None:
        self.qi = (float(qi[0]), float(qi[1]), float(qi[2]))
        self.type = int(path_type)
        self.param = [float(p) for p in params]
        self.rho = float(rho)

    # ------------------------------------------------------------------
    # Path inspection methods
    def path_type(self) -> int:
        """Return the integer identifying the path type.

        Returns
        -------
        int
            Path type in ``0..5``.
        """
        return self.type

    def path_length(self) -> float:
        """Total length of the path in world units."""
        return sum(self.param) * self.rho

    def segment_length(self, i: int) -> float:
        """Length of segment ``i`` in world units.

        Parameters
        ----------
        i : int
            Index of the segment (0, 1 or 2).
        """
        if not 0 <= i <= 2:
            raise IndexError("segment index out of range: {}".format(i))
        return self.param[i] * self.rho

    def segment_length_normalized(self, i: int) -> float:
        """Length of segment ``i`` normalised by ``rho``.

        Parameters
        ----------
        i : int
            Index of the segment (0, 1 or 2).
        """
        if not 0 <= i <= 2:
            raise IndexError("segment index out of range: {}".format(i))
        return self.param[i]

    # ------------------------------------------------------------------
    # Sampling methods
    def sample(self, t: float) -> Tuple[float, float, float]:
        """Sample the configuration at distance ``t`` along the path.

        Parameters
        ----------
        t : float
            Distance along the path from the start configuration.  Must
            satisfy ``0 <= t <= path_length``.

        Returns
        -------
        tuple of float
            A configuration ``(x, y, heading)`` at the requested distance.
        """
        path_len = self.path_length()
        if t < 0.0 or t > path_len:
            raise ValueError(
                f"sample distance {t} outside path length [0, {path_len}]"
            )
        # Normalise distance by rho
        tprime = t / self.rho
        p1, p2, p3 = self.param
        seg_types = _SEGMENT_SEQUENCES[self.type]
        # Build intermediate targets after segment 1 and 2
        # Start orientation at origin
        qi_x, qi_y, qi_th = 0.0, 0.0, self.qi[2]
        # Compute intermediate end of segment1 (q1) and segment2 (q2)
        q1 = _dubins_segment(p1, (qi_x, qi_y, qi_th), seg_types[0])
        q2 = _dubins_segment(p2, q1, seg_types[1])
        # Determine which segment we are in and compute local sample
        if tprime < p1:
            # In first segment
            q = _dubins_segment(tprime, (qi_x, qi_y, qi_th), seg_types[0])
        elif tprime < (p1 + p2):
            # In second segment
            q = _dubins_segment(tprime - p1, q1, seg_types[1])
        else:
            # In third segment
            q = _dubins_segment(tprime - p1 - p2, q2, seg_types[2])
        # Scale back to world units and translate back to start position
        x = q[0] * self.rho + self.qi[0]
        y = q[1] * self.rho + self.qi[1]
        th = mod2pi(q[2])
        return x, y, th

    def sample_many(self, step_size: float) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """Sample the path at regular intervals.

        Parameters
        ----------
        step_size : float
            Step size along the path in world units.  Must be positive.

        Returns
        -------
        tuple
            A tuple ``(configurations, distances)``.  ``configurations`` is a
            list of ``(x, y, heading)`` tuples sampled along the path and
            ``distances`` is a list of the corresponding path distances.
        """
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        result_configs: List[Tuple[float, float, float]] = []
        result_dists: List[float] = []
        length = self.path_length()
        d = 0.0
        while d < length:
            result_configs.append(self.sample(d))
            result_dists.append(d)
            d += step_size
        # Always include the end configuration
        if not math.isclose(d, length, rel_tol=1e-12, abs_tol=1e-12):
            result_configs.append(self.sample(length))
            result_dists.append(length)
        return result_configs, result_dists

    def extract_subpath(self, t: float) -> 'DubinsPath':
        """Return a subpath consisting of the first ``t`` units of this path.

        Parameters
        ----------
        t : float
            Length of the subpath to extract.  Must satisfy ``0 <= t <= path_length``.

        Returns
        -------
        DubinsPath
            A new ``DubinsPath`` representing the initial portion of this path.
        """
        if t < 0.0 or t > self.path_length():
            raise ValueError(
                f"subpath length {t} outside [0, {self.path_length()}]"
            )
        # Normalise t
        tprime = t / self.rho
        new_params = [0.0, 0.0, 0.0]
        remaining = tprime
        for i in range(3):
            segment = min(self.param[i], remaining)
            new_params[i] = segment
            remaining -= segment
        return DubinsPath(self.qi, self.type, new_params, self.rho)

    def path_endpoint(self) -> Tuple[float, float, float]:
        """Return the final configuration of the path."""
        return self.sample(self.path_length())

    # Provide representation for debugging
    def __repr__(self) -> str:
        params_str = ', '.join(f"{p:.3f}" for p in self.param)
        return f"<DubinsPath type={self.type} params=[{params_str}] rho={self.rho}>"


def shortest_path(q0: Tuple[float, float, float], q1: Tuple[float, float, float],
                  rho: float) -> DubinsPath:
    """Compute the shortest Dubins path between two configurations.

    Parameters
    ----------
    q0 : tuple of float
        Start configuration ``(x, y, heading)``.
    q1 : tuple of float
        End configuration ``(x, y, heading)``.
    rho : float
        Minimum turning radius of the vehicle.  Must be positive.

    Returns
    -------
    DubinsPath
        A ``DubinsPath`` object representing the shortest path.
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    # Precompute intermediate results
    ir = _intermediate_results(q0, q1, rho)
    best_cost = math.inf
    best_type: Optional[int] = None
    best_params: Optional[Tuple[float, float, float]] = None
    # Iterate over all path types
    for path_type in (LSL, LSR, RSL, RSR, RLR, LRL):
        params = _dubins_word(ir, path_type)
        if params is not None:
            cost = params[0] + params[1] + params[2]
            if cost < best_cost:
                best_cost = cost
                best_type = path_type
                best_params = params
    if best_type is None or best_params is None:
        raise RuntimeError("no valid Dubins path found between given configurations")
    return DubinsPath(q0, best_type, best_params, rho)


def path(q0: Tuple[float, float, float], q1: Tuple[float, float, float],
         rho: float, path_type: int) -> Optional[DubinsPath]:
    """Compute a Dubins path of a specific type between two configurations.

    This function mirrors the behaviour of ``dubins.path`` in the original
    library.  It returns ``None`` if the requested path type does not
    yield a valid path.

    Parameters
    ----------
    q0 : tuple of float
        Start configuration ``(x, y, heading)``.
    q1 : tuple of float
        End configuration ``(x, y, heading)``.
    rho : float
        Minimum turning radius.  Must be positive.
    path_type : int
        Integer in ``0..5`` specifying the candidate path type.

    Returns
    -------
    DubinsPath or None
        The resulting path or ``None`` if no path of that type exists.
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if path_type not in (LSL, LSR, RSL, RSR, RLR, LRL):
        raise ValueError(f"invalid path type {path_type}")
    ir = _intermediate_results(q0, q1, rho)
    params = _dubins_word(ir, path_type)
    if params is None:
        return None
    return DubinsPath(q0, path_type, params, rho)


def path_sample(q0: Tuple[float, float, float], q1: Tuple[float, float, float],
                rho: float, step_size: float) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """Generate a sampled Dubins path between two configurations.

    This convenience function constructs the shortest Dubins path and
    samples it at regular intervals.  It returns a list of
    configurations and the corresponding path distances.  It is
    equivalent to calling ``shortest_path(q0, q1, rho).sample_many(step_size)``.

    Parameters
    ----------
    q0 : tuple of float
        Start configuration ``(x, y, heading)``.
    q1 : tuple of float
        End configuration ``(x, y, heading)``.
    rho : float
        Minimum turning radius.  Must be positive.
    step_size : float
        Step size along the path in world units.  Must be positive.

    Returns
    -------
    tuple
        ``(configurations, distances)`` where ``configurations`` is a
        list of sampled configurations and ``distances`` are the
        corresponding path distances.
    """
    path_obj = shortest_path(q0, q1, rho)
    return path_obj.sample_many(step_size)


# ----------------------------------------------------------------------
# Internal helper functions (not exposed in __all__)

def _intermediate_results(q0: Tuple[float, float, float], q1: Tuple[float, float, float],
                          rho: float) -> dict:
    """Compute intermediate values used by the Dubins path equations."""
    x0, y0, th0 = q0
    x1, y1, th1 = q1
    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy)
    d = D / rho
    # Avoid domain errors if the poses coincide
    theta = mod2pi(math.atan2(dy, dx)) if d > 0.0 else 0.0
    alpha = mod2pi(th0 - theta)
    beta = mod2pi(th1 - theta)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)
    return {
        'alpha': alpha,
        'beta': beta,
        'd': d,
        'sa': sa,
        'sb': sb,
        'ca': ca,
        'cb': cb,
        'c_ab': c_ab,
        'd_sq': d * d,
    }


def _dubins_word(ir: dict, path_type: int) -> Optional[Tuple[float, float, float]]:
    """Compute the normalised segment parameters for a given path type.

    Returns ``None`` if the path type yields no valid solution.
    """
    if path_type == LSL:
        return _dubins_LSL(ir)
    elif path_type == LSR:
        return _dubins_LSR(ir)
    elif path_type == RSL:
        return _dubins_RSL(ir)
    elif path_type == RSR:
        return _dubins_RSR(ir)
    elif path_type == RLR:
        return _dubins_RLR(ir)
    elif path_type == LRL:
        return _dubins_LRL(ir)
    else:
        return None


def _dubins_LSL(ir: dict) -> Optional[Tuple[float, float, float]]:
    tmp0 = ir['d'] + ir['sa'] - ir['sb']
    p_sq = 2 + ir['d_sq'] - 2 * ir['c_ab'] + 2 * ir['d'] * (ir['sa'] - ir['sb'])
    if p_sq >= 0.0:
        tmp1 = math.atan2(ir['cb'] - ir['ca'], tmp0)
        t = mod2pi(tmp1 - ir['alpha'])
        p = math.sqrt(p_sq)
        q = mod2pi(ir['beta'] - tmp1)
        return (t, p, q)
    return None


def _dubins_RSR(ir: dict) -> Optional[Tuple[float, float, float]]:
    tmp0 = ir['d'] - ir['sa'] + ir['sb']
    p_sq = 2 + ir['d_sq'] - 2 * ir['c_ab'] + 2 * ir['d'] * (ir['sb'] - ir['sa'])
    if p_sq >= 0.0:
        tmp1 = math.atan2(ir['ca'] - ir['cb'], tmp0)
        t = mod2pi(ir['alpha'] - tmp1)
        p = math.sqrt(p_sq)
        q = mod2pi(tmp1 - ir['beta'])
        return (t, p, q)
    return None


def _dubins_LSR(ir: dict) -> Optional[Tuple[float, float, float]]:
    p_sq = -2 + ir['d_sq'] + 2 * ir['c_ab'] + 2 * ir['d'] * (ir['sa'] + ir['sb'])
    if p_sq >= 0.0:
        p = math.sqrt(p_sq)
        # Note: break into two atan2 calls to replicate the C code's difference of angles
        tmp2 = math.atan2(-ir['ca'] - ir['cb'], ir['d'] + ir['sa'] + ir['sb'])
        tmp1 = math.atan2(-2.0, p)
        tmp0 = tmp2 - tmp1
        t = mod2pi(tmp0 - ir['alpha'])
        q = mod2pi(tmp0 - mod2pi(ir['beta']))
        return (t, p, q)
    return None


def _dubins_RSL(ir: dict) -> Optional[Tuple[float, float, float]]:
    p_sq = -2 + ir['d_sq'] + 2 * ir['c_ab'] - 2 * ir['d'] * (ir['sa'] + ir['sb'])
    if p_sq >= 0.0:
        p = math.sqrt(p_sq)
        tmp2 = math.atan2(ir['ca'] + ir['cb'], ir['d'] - ir['sa'] - ir['sb'])
        tmp1 = math.atan2(2.0, p)
        tmp0 = tmp2 - tmp1
        t = mod2pi(ir['alpha'] - tmp0)
        q = mod2pi(ir['beta'] - tmp0)
        return (t, p, q)
    return None


def _dubins_RLR(ir: dict) -> Optional[Tuple[float, float, float]]:
    tmp0 = (6.0 - ir['d_sq'] + 2 * ir['c_ab'] + 2 * ir['d'] * (ir['sa'] - ir['sb'])) / 8.0
    # Guard against floating point overshoot
    if abs(tmp0) <= 1.0:
        phi = math.atan2(ir['ca'] - ir['cb'], ir['d'] - ir['sa'] + ir['sb'])
        p = mod2pi(2 * math.pi - math.acos(max(-1.0, min(1.0, tmp0))))
        t = mod2pi(ir['alpha'] - phi + mod2pi(p / 2.0))
        q = mod2pi(ir['alpha'] - ir['beta'] - t + mod2pi(p))
        return (t, p, q)
    return None


def _dubins_LRL(ir: dict) -> Optional[Tuple[float, float, float]]:
    tmp0 = (6.0 - ir['d_sq'] + 2 * ir['c_ab'] + 2 * ir['d'] * (ir['sb'] - ir['sa'])) / 8.0
    if abs(tmp0) <= 1.0:
        phi = math.atan2(ir['ca'] - ir['cb'], ir['d'] + ir['sa'] - ir['sb'])
        p = mod2pi(2 * math.pi - math.acos(max(-1.0, min(1.0, tmp0))))
        t = mod2pi(-ir['alpha'] - phi + p / 2.0)
        q = mod2pi(mod2pi(ir['beta']) - ir['alpha'] - t + mod2pi(p))
        return (t, p, q)
    return None


def _dubins_segment(t: float, qi: Tuple[float, float, float], segment_type: str) -> Tuple[float, float, float]:
    """Follow a segment of a given type for a normalised distance ``t``.

    This function is an internal helper used by :meth:`DubinsPath.sample`.
    It assumes the starting pose ``qi`` is given in the rotated frame
    where the initial position is at the origin.  The returned pose
    remains in that rotated frame and must be scaled by ``rho`` and
    translated back to world coordinates by the caller.

    Parameters
    ----------
    t : float
        Normalised distance along the segment (i.e. divided by ``rho``).
    qi : tuple of float
        Start pose in the rotated frame.
    segment_type : str
        One of ``'L'``, ``'R'`` or ``'S'``.

    Returns
    -------
    tuple of float
        The pose ``(x, y, heading)`` after travelling distance ``t`` along
        the specified segment.
    """
    x0, y0, th0 = qi
    if segment_type == 'L':
        # Left turn: rotate CCW by t
        new_x = x0 + math.sin(th0 + t) - math.sin(th0)
        new_y = y0 - math.cos(th0 + t) + math.cos(th0)
        new_th = th0 + t
    elif segment_type == 'R':
        # Right turn: rotate CW by t
        new_x = x0 - math.sin(th0 - t) + math.sin(th0)
        new_y = y0 + math.cos(th0 - t) - math.cos(th0)
        new_th = th0 - t
    elif segment_type == 'S':
        # Straight segment: move forward in current heading
        new_x = x0 + math.cos(th0) * t
        new_y = y0 + math.sin(th0) * t
        new_th = th0
    else:
        raise ValueError(f"invalid segment type {segment_type}")
    return (new_x, new_y, new_th)


__all__ = [
    'DubinsPath',
    'shortest_path',
    'path',
    'path_sample',
    'LSL',
    'LSR',
    'RSL',
    'RSR',
    'RLR',
    'LRL',
]