import sys
import pickle
import multiprocessing
import multiprocessing.shared_memory
from multiprocessing import resource_tracker

import numpy as np

def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]

class SMArray:
    """
    Sending numpy arrays across process boundaries incurs serialization/deserializatoin + copy overhead
    Using shared memory can reduce this serialization and copy overhead
    This class allocates a block of shared memory that can be accessed by multiple processes without
    costly copies & serialization/deserialization
    """
    def __init__(self, shape, dtype):
        if not isinstance(shape, (list, tuple)):
            raise TypeError("shape must be a list/tuple")
        if not ((isinstance(dtype, type) and issubclass(dtype, np.number)) or isinstance(dtype, np.dtype)):
            raise TypeError("dtype must be a np.dtype")
        if isinstance(dtype, type):
            dtype = dtype()
        self._shape = shape
        self._dtype = dtype
        remove_shm_from_resource_tracker()
        num_bytes = np.prod(shape) * dtype.itemsize
        self._shm = multiprocessing.shared_memory.SharedMemory(size=num_bytes, create=True)
        self._ndarray = np.ndarray(shape, dtype, buffer=self._shm.buf)
        self._freed = False

    @classmethod
    def from_ndarray(cls, ndarray):
        arr = cls(ndarray.shape, ndarray.dtype)
        arr.as_ndarray()[:] = ndarray[:]
        return arr

    def as_ndarray(self):
        return self._ndarray

    def free(self):
        if self._freed:
            return
        self._shm.close()
        self._shm.unlink()
        self._freed = True

    def __getstate__(self):
        return dict(shape=self._shape, dtype=self._dtype, shm_id=self._shm.name, freed=self._freed)

    def __setstate__(self, state):
        remove_shm_from_resource_tracker()
        self._shape = state['shape']
        self._dtype = state['dtype']
        if not state['freed']:
            self._shm = multiprocessing.shared_memory.SharedMemory(name=state['shm_id'])
        else:
            num_bytes = np.prod(self._shape) * self._dtype.itemsize
            self._shm = multiprocessing.shared_memory.SharedMemory(create=True, size=num_bytes)
        self._ndarray = np.ndarray(self._shape, self._dtype, buffer=self._shm.buf)
        self._freed = False

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    x = np.random.randn(2560, 1440)
    N = 100
    if len(sys.argv) > 1 and sys.argv[1] == 'shm':
        for i in range(N):
            print(i)
            sx = SMArray.from_ndarray(x)
            queue.put(sx)
            y = queue.get()
            assert np.all(np.equal(x, y.as_ndarray()))
            y.free()
    else:
        for i in range(N):
            print(i)
            queue.put(x)
            y = queue.get()
            assert np.all(np.equal(x, y))

