# Minimal decorator
import time
from functools import wraps

import numpy as np


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time for '{func.__name__}': {elapsed_time:.6f} seconds")
        return result, elapsed_time

    return wrapper


@measure_time
def numpy_sum(x):
    return np.sum(x)


@measure_time
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
