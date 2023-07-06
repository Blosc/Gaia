# Count the number of starts in the 3d array by getting metainfo and decompress chunks only when needed
# As a bonus, the maximum number of stars in a cell is also computed.

import sys
import blosc2
import numpy as np


def get_stats(fname):
    data = blosc2.open(fname)
    print(f"{fname=} has {data.schunk.nchunks} chunks and {data.shape} shape")
    chunk = np.empty(np.prod(data.chunks), dtype=data.dtype).reshape(data.chunks)
    nstars_, std_, min_, max_ = 0, 0, 1, 0
    nchunks = 0
    nsat = 0
    for info in data.iterchunks_info():
        if info.special == blosc2.SpecialValue.NOT_SPECIAL:
            data.schunk.decompress_chunk(info.nchunk, dst=chunk)
            # nstars2, std2, min2, max2 = chunk.sum(), chunk.std(), chunk.min(), chunk.max()
            nstars2, max2 = chunk.sum(), chunk.max()
            if max2 == 255:
                nsat += 1
            nstars_ += nstars2
            nchunks += 1
            #std_ += std2
            #min_ = min(min_, min2)
            max_ = max(max_, max2)
    print(f"{nstars_=} ({nchunks / data.schunk.nchunks:.4f} occupancy)")
    #print(f"{nstars_=} ({std_ / data.schunk.nchunks:.4f})")
    print(f"{max_=}, {nsat=}")

fname = sys.argv[1]
get_stats(fname)
