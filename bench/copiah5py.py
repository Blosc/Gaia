import blosc2
import numpy as np
from time import time
import h5py
import hdf5plugin
import os


# path = "/home/blosc/gaia/gaia-3d-windows.b2nd"
# arr = blosc2.open(path)
path_blosclz = "/home/marta/BTune-training/analysis/copiagaia0.b2nd"
arr = blosc2.open(path_blosclz)
print(arr.info)


clevel = 5
cname = "blosclz"
fname_h5py = 'h5pycopy-blosc1.hdf5'

t0 = time()
filters = hdf5plugin.Blosc(clevel=clevel, cname=cname, shuffle=hdf5plugin.Blosc.NOSHUFFLE)
# tables.set_blosc_max_threads(nthreads)
h5pyf = h5py.File(fname_h5py, "w")
h5d = h5pyf.create_dataset("dataset", dtype=arr.dtype, shape=arr.shape, chunks=arr.chunks, **filters)

for info in arr.iterchunks_info():
    x = info.coords[0]
    y = info.coords[1]
    z = info.coords[2]
    if info.nchunk <= 2:
        print(x, y, z)

    h5d[x * arr.chunks[0]:(x + 1) * arr.chunks[0], y * arr.chunks[0]:(y + 1) * arr.chunks[0],
    z * arr.chunks[0]:(z + 1) * arr.chunks[0]] = arr[x * arr.chunks[0]:(x + 1) * arr.chunks[0],
                                                 y * arr.chunks[0]:(y + 1) * arr.chunks[0],
                                                 z * arr.chunks[0]:(z + 1) * arr.chunks[0]]
t = time() - t0

dset_size = np.prod(arr.shape) * arr.dtype.itemsize
speed = dset_size / (t * 2**30)
num_blocks = os.stat(fname_h5py).st_blocks
# block_size = os.statvfs(fname_h5py).f_bsize
size_on_disk = num_blocks * 512
cratio = dset_size / size_on_disk
print(f"Time for filling array (hdf5, h5py): {t:.3f} s ({speed:.2f} GB/s) ; cratio: {cratio:.1f}x")

h5pyf.close()
