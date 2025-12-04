# Advanced decorator
import time
from functools import wraps

import numpy as np

profiling_data = {}


def profile(name=None):
    if callable(name):
        fn = name
        tag = fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            dt = time.perf_counter() - t0

            entry = profiling_data.setdefault(tag, {"n": 0, "total": 0.0})
            entry["n"] += 1
            entry["total"] += dt
            return out

        return wrapper

    def decorator(fn):
        tag = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            dt = time.perf_counter() - t0

            entry = profiling_data.setdefault(tag, {"n": 0, "total": 0.0})
            entry["n"] += 1
            entry["total"] += dt
            return out

        return wrapper

    return decorator


def get_profiling_stats():
    stats = {}
    for name, d in profiling_data.items():
        n = d["n"]
        total = d["total"]
        stats[name] = {
            "n_calls": n,
            "total": total,
            "mean": total / n if n else 0.0,
        }
    return stats


@profile
def numpy_sum(x):
    return np.sum(x)


@profile
def python_sum(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]
    return sum


# Test
np_x = np.ones(10000000)
list_x = list(np_x.copy())

numpy_sum(np_x)
python_sum(list_x)
print(get_profiling_stats())
