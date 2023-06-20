# Count the number of starts in the 3d array by getting metainfo and decompress chunks only when needed
# As a bonus, the maximum number of stars in a cell is also computed.

import sys
import blosc2
import numpy as np
import tables as tb


def get_stats(b2file, h5file):
    data = blosc2.open(b2file)
    h5f = tb.open_file(h5file)
    table = h5f.root.data
    chunk = np.empty(np.prod(data.chunks), dtype=data.dtype).reshape(data.chunks)
    nstars_, std_, min_, max_ = 0, 0, 1, 0
    nchunks = 0
    for info in data.iterchunks_info():
        if info.special == blosc2.SpecialValue.NOT_SPECIAL:
            data.schunk.decompress_chunk(info.nchunk, dst=chunk)
            # nstars2, std2, min2, max2 = chunk.sum(), chunk.std(), chunk.min(), chunk.max()
            nstars2, max2 = chunk.sum(), chunk.max()
            nstars_ += nstars2
            nchunks += 1
            if max2 >= 1:
                idx = chunk.argmax()
                cell_coords_in_chunk = np.unravel_index(idx, data.chunks)
                cell_coords = np.array(info.coords) * np.array(data.chunks) + cell_coords_in_chunk
                print(f"{max2=} {info.coords=}, {cell_coords=}")
                idx_meta = cell_coords[0] * data.shape[1] * data.shape[2] + cell_coords[1] * data.shape[2] + \
                           cell_coords[2]
                meta = list(table.where("(idx < idx_meta - 10) & (idx <= idx_meta + 10)"))
                print(f"{meta=}")
            #std_ += std2
            #min_ = min(min_, min2)
            max_ = max(max_, max2)
    print(f"{nstars_=} ({nchunks / data.schunk.nchunks:.4f} occupancy)")
    #print(f"{nstars_=} ({std_ / data.schunk.nchunks:.4f})")
    print(f"{max_=}")

b2file, h5file  = sys.argv[1:]
get_stats(b2file, h5file)
