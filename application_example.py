import time
import multiprocessing
import queue
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from sm_array import SMArray

class PicklableVideoCapture:
    """
    cv.VideoCapture doesn't define a __dict__ attribute
    which makes it un-picklable
    This wrapper class provides most of cv.VideoCapture's interface
    while defining how to serialize/deserialize it using the pickle
    module
    """
    def __init__(self, filename):
        self._fname = filename
        self._cap = cv.VideoCapture(filename)

    def get(self, propId):
        return self._cap.get(propId)

    def getBackendName(self):
        return self._cap.getBackendName()

    def getExceptionMode(self):
        return self._cap.getExceptionMode()

    def grab(self):
        return self._cap.grab()

    def isOpened(self):
        return self._cap.isOpened()

    def open(self, filename):
        return self._cap.open(filename)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()

    def __getstate__(self):
        state = dict(filename=self._fname)
        for prop in ['CAP_PROP_POS_MSEC', 'CAP_PROP_POS_FRAMES']:
            state[prop] = self._cap.get(getattr(cv, prop))
        return state

    def __setstate__(self, state):
        self._fname = state.pop('filename')
        self._cap = cv.VideoCapture(self._fname)
        for prop in state:
            self._cap.set(getattr(cv, prop), state[prop])


class ImageProcClass:
    """
    Toy example class that processes multiple videos in parallel using Processes
    Each process sends its results to the main process using a Queue
    Sending data across process boundaries incurs serialization/deserializatoin + copy overhead
    This exmaple demonstrates that replacing np.ndarrays with SMArrays results in a speedup
    """
    def __init__(self, paths, output_height, output_width, worker_count=1, use_smarray=False):
        self._worker_count = worker_count
        self._use_smarray = use_smarray
        self._paths = paths
        self._sources = []
        for p in paths:
            cap = PicklableVideoCapture(str(p))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open: {p}")
            self._sources.append(cap)
        self._img_queue = multiprocessing.Queue()
        self._task_queue = multiprocessing.Queue()
        self._workers = []
        self._out_width = output_width
        self._out_height = output_height

    def _process(self, arr):
        # some processing here
        return cv.resize(arr, (self._out_width, self._out_height))

    def _load_image(self, source):
        ok, img = source.read()
        if not ok:
            return None
        img = self._process(img)
        if self._use_smarray:
            return SMArray.from_ndarray(img)
        return img

    def _worker_fn(self, task_queue, output_queue, done_event):
        put_times = []
        while not done_event.is_set():
            try:
                source = task_queue.get(timeout=0.5)
            except queue.Empty:
                print('Avg frame serialize time', np.round(np.mean(put_times)*1000, 2), 'ms')
                return
            while True:
                out = self._load_image(source)
                if out is None:
                    output_queue.put(None)
                    break
                t1 = time.time()
                output_queue.put(out)
                put_times.append(time.time() - t1)

    def _prepare_work(self):
        for source in self._sources:
            self._task_queue.put(source)

    def _prepare_workers(self):
        self._done_event = multiprocessing.Event()
        for i in range(self._worker_count):
            proc = multiprocessing.Process(target=self._worker_fn, args=(self._task_queue, self._img_queue, self._done_event))
            proc.start()
            self._workers.append(proc)

    def _collect_outputs(self):
        done_count = 0
        frames_done = 0
        get_times = []
        start_time = time.time() 
        while done_count < len(self._sources):
            t1 = time.time()
            out = self._img_queue.get()
            get_times.append(time.time() - t1)
            if out is None:
                done_count += 1
                continue
            frames_done += 1
            if self._use_smarray:
                out.free()

        total_time = time.time() - start_time
        print('Avg frame deserialize time', np.round(np.mean(get_times)*1000, 2), 'ms')
        print(f'Avg FPS: {np.round(frames_done/total_time, 2)}')
        self._done_event.set()
        [proc.join() for proc in self._workers]
        [source.release() for source in self._sources]

    def run(self):
        self._prepare_work()
        self._prepare_workers()
        self._collect_outputs()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An example application that demos speedup using SMArrays")
    parser.add_argument("sources", type=str, nargs='+', help="video/camera paths")
    parser.add_argument("--smarray", dest='use_smarray', action='store_true', help="Wrap np.ndarrays into SMArrays before sending")
    parser.add_argument("--out-height", dest='output_height', type=int, default=1920, help="Height of the resized output frames")
    parser.add_argument("--out-width", dest='output_width', type=int, default=1080, help="Width of the resized output frames")
    parser.add_argument("-j", dest='worker_count', type=int, default=1, help="number of worker processes")
    args = parser.parse_args()
    sources = [Path(s).expanduser().resolve() for s in args.sources]
    ic = ImageProcClass(sources, worker_count=args.worker_count, output_height=args.output_height, output_width=args.output_width, use_smarray=args.use_smarray)
    ic.run()
