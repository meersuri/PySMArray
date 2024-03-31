import multiprocessing

import numpy as np
from sm_array import SMArray

def sm_arr_worker_fn(sm_arr):
    arr = sm_arr.as_ndarray()
    arr /= arr.max() # modify in-place

def nd_arr_worker_fn(arr):
    arr /= arr.max() # modify in-place

def nd_arr_main():
    arr = 10*np.random.rand(1280,720,3) # rand values in the range [0, 10)
    print("Max val:", arr.max()) # modification done in worker process NOT reflected in parent process
    worker = multiprocessing.Process(target=nd_arr_worker_fn, args=(arr,))
    worker.start()
    worker.join()
    print("Max val:", arr.max())

def sm_arr_main():
    arr = 10*np.random.rand(1280,720,3) # rand values in the range [0, 10)
    sm_arr = SMArray.from_ndarray(arr)
    print("Max val:", sm_arr.as_ndarray().max())
    sm_arr.free()
    worker = multiprocessing.Process(target=sm_arr_worker_fn, args=(sm_arr,))
    worker.start()
    worker.join()
    print("Max val:", sm_arr.as_ndarray().max()) # modification done in worker process reflects in parent process

if __name__ == '__main__':
    sm_arr_main()

