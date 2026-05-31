"""Shared wall-clock budget for FD loops (set by the optimization methods).

FD callables built outside a method (lambdas that don't thread t_start/
time_limit) read this module-level budget so they still honour the limit.
A list is used so importers see element mutations.
"""


class TimeLimitExceeded(Exception):
    """Raised when the wall-clock budget is exceeded inside an FD loop."""


_time_budget = [None, None]   # [t_start, time_limit]


def set_time_budget(t_start, time_limit):
    _time_budget[0] = t_start
    _time_budget[1] = time_limit


def clear_time_budget():
    _time_budget[0] = None
    _time_budget[1] = None
