import time
import scipy.signal
import argparse
import numpy as np

def run(size):
    """ Run a fftconvolve test run and time it. 
        Dummy values are used in this script since this is a quick latency test only. 
        This aims to test on nano-2. """
    src    = np.random.randint(0, 16, (size, size)) 
    kernel = np.random.randint(0, 16, (7, 6)) 

    start = time.time()
    ret   = scipy.signal.fftconvolve(src, kernel, mode='same')
    end   = time.time()

    elapsed = end - start
    print("scipy.signal.fftconvolve latency: ", elapsed * 1000.0, " (ms) with problem size = ", size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', action='store', required=True, type=int, help="problem size of the FFT-based convolution.")
    args = parser.parse_args()
    size = args.size
    run(size)
