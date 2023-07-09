import blosc2
import numpy as np
from time import time
import h5py
import hdf5plugin
import tables


# path = "/home/blosc/gaia/gaia-3d-windows.b2nd"
# arr = blosc2.open(path)
path_blosclz = "/home/marta/BTune-training/analysis/copiagaia0.b2frame"
arr = blosc2.open(path_blosclz)
print(arr.info)


clevel = 5
cname = "blosclz"
fname_tables = 'tablescopy.hdf5'

t0 = time()
filters = tables.Filters(complevel=clevel, complib="blosc:%s" % cname, shuffle=False)
# tables.set_blosc_max_threads(nthreads)
h5f = tables.open_file(fname_tables, "w")
h5ca = h5f.create_carray(h5f.root, "carray", filters=filters, chunkshape=arr.chunks, shape=arr.shape, atom=arr.dtype)

for info in arr.interchunks_info():
    x = info.coords[0]
    y = info.coords[1]
    z = info.coords[2]
    if info.nchunk <= 2:
        print(x, y, z)

    h5ca[x * arr.chunks[0]:(x + 1) * arr.chunks[0], y * arr.chunks[0]:(y + 1) * arr.chunks[0],
    z * arr.chunks[0]:(z + 1) * arr.chunks[0]] = arr[x * arr.chunks[0]:(x + 1) * arr.chunks[0],
                                                 y * arr.chunks[0]:(y + 1) * arr.chunks[0],
                                                 z * arr.chunks[0]:(z + 1) * arr.chunks[0]]
t = time() - t0

dset_size = np.prod(arr.shape) * arr.dtype.itemsize
speed = dset_size / (t * 2**30)
cratio = dset_size / h5ca.size_on_disk
print(f"Time for filling array (hdf5, tables): {t:.3f} s ({speed:.2f} GB/s) ; cratio: {cratio:.1f}x")

h5f.close()

