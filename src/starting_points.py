"""Starting point generation for experiments.

Specification from the assignment:
    - 1 suggested point (x_bar) + 5 random points sampled uniformly in
      [x_bar_i - 1, x_bar_i + 1] for each component i.
    - seed = minimum student-ID in the team (set ONCE before generating
      the points for all problems/dimensions, to ensure reproducibility).
"""
import numpy as np


def generate_starting_points(x_bar, num_random=5, rng=None):
    """Return a list of starting points: [x_bar, x_bar + delta_1, ...].

    Parameters
    ----------
    x_bar      : ndarray (n,) — suggested starting point
    num_random : number of additional random points (default 5)
    rng        : numpy.random.Generator (optional).
                 If None, uses the global np.random.
    """
    x_bar = np.asarray(x_bar, dtype=float)
    n = len(x_bar)
    points = [x_bar.copy()]
    sample = rng.uniform if rng is not None else np.random.uniform
    for _ in range(num_random):
        delta = sample(-1.0, 1.0, size=n)
        points.append(x_bar + delta)
    return points
