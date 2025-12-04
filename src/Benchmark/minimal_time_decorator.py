# Minimal decorator
import time
from functools import wraps

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
