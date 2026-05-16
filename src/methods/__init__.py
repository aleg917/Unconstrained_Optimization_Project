"""Metodi di ottimizzazione."""
from .steepest_descent import steepest_descent
from .armijo_backtracking import armijo_backtracking
from .modified_newton import modified_newton

__all__ = ["steepest_descent", "armijo_backtracking", "modified_newton"]
