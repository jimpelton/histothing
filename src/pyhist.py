import time
import argparse
import sys

import numpy as np
import numba
from numba import jit, njit, config, threading_layer


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="", help="Input file")
    parser.add_argument("--out", default="", help="Output file")
    parser.add_argument("--dtype", default="i8", help="Data type")

    return parser.parse_args()

@njit(fastmath=True)
def vol_min_max(in_file):
    n = len(in_file)
    vol_min = np.iinfo(fd.dtype).max
    vol_max = -np.iinfo(fd.dtype).min
    for i in numba.range(n):
        x = fd[i]
        if x < vol_min:
            vol_min = x
        if x > vol_max:
            vol_max = x

    return vol_min, vol_max

@njit(fastmath=True)
def dohist(in_file, vol_min, vol_max, num_bins):
    n = len(in_file)
    max_idx = numba.int64(num_bins - 1)
    histcount = np.zeros(num_bins)

    for i in numba.prange(n):
        x = fd[i]
        idx = numba.int64((x - vol_min) / (vol_max - vol_min) * numba.float64(max_idx) + 0.5)
        if idx > max_idx:
            idx = max_idx

        histcount[idx] += 1

    # compute percentage of values in each bin
    for i in numba.prange(len(histcount)):
        histcount[i] = histcount[i] / n

def main(cargs):
    fd = np.memmap(cargs.file, dtype=np.dtype(cargs.dtype), mode='r')
    vol_min, vol_max = vol_min_max(fd)
    num_bins = 1536
    hist = dohist(fd, vol_min, vol_max, num_bins)
    
    np.savetxt(cargs.out, bins)

if __name__ == '__main__':
    cargs = parse_args(sys.argv[1:])
    main(cargs)
