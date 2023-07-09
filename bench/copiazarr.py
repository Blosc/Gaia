import blosc2
import numpy as np
import zarr
from numcodecs import Blosc


path = "/home/blosc/gaia/gaia-3d-windows.b2nd"
# arr = blosc2.open(path)
path_blosclz = "/home/marta/BTune-training/analysis/copiagaia0.b2frame"
arr = blosc2.open(path_blosclz)

compressor = Blosc(cname='blosclz', clevel=5, shuffle=Blosc.NOSHUFFLE)
bzarr = zarr.open('zarrcopia', mode='w', shape=arr.shape,
               chunks=arr.chunks, dtype=arr.dtype, compressor=compressor)
for info in arr.iterchunks_info():
    x = info.coords[0]
    y = info.coords[1]
    z = info.coords[2]
    if info.nchunk<=2:
        print(x, y, z)
    
    bzarr[x*arr.chunks[0]:(x+1)*arr.chunks[0], y*arr.chunks[0]:(y+1)*arr.chunks[0], z*arr.chunks[0]:(z+1)*arr.chunks[0]] = arr[x*arr.chunks[0]:(x+1)*arr.chunks[0], y*arr.chunks[0]:(y+1)*arr.chunks[0], z*arr.chunks[0]:(z+1)*arr.chunks[0]]

print("end")

