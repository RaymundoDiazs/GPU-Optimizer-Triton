import time

def benchmark(function, *args):
    start = time.time()
    result = function(*args)
    end = time.time()
    return {
        "execution_time": end - start,
        "result": result
    }
