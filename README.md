`SMArray` is wrapper around `np.ndarray` that is backed by Python `multiprocessing` shared memory. This has the following advantages -  
- Cheap serialization/deserialization across process boundaries
- Modifications by child processes reflect in the parent process. 
```
import multiprocessing

import numpy as np
from sm_array import SMArray

def sm_arr_worker_fn(sm_arr):
    arr = sm_arr.as_ndarray()
    arr /= arr.max() # modify in-place

if __name__ == '__main__':
    arr = 10*np.random.rand(1280,720,3) # rand values in the range [0, 10)
    sm_arr = SMArray.from_ndarray(arr)
    print("Max val:", sm_arr.as_ndarray().max())
    sm_arr.free()
    worker = multiprocessing.Process(target=sm_arr_worker_fn, args=(sm_arr,))
    worker.start()
    worker.join()
    print("Max val:", sm_arr.as_ndarray().max()) # modification done in worker process reflects in parent process
```
