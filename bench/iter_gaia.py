import blosc2
import numpy as np


path = "/home/blosc/gaia/gaia-3d-windows.b2nd"
arr = blosc2.open(path)

storage = {"urlpath": "gaiacopia.b2frame"}
barr = blosc2.SChunk(chunksize=arr.schunk.chunksize, **storage)
print(barr.chunksize)

print(arr.info)

print("nchunks ", arr.schunk.nchunks)

for info in arr.iterchunks_info():
    print(info.nchunk)
    if info.nchunk % 53 == 0:
        barr.append_data(arr.schunk.decompress_chunk(info.nchunk))

print("nchunks copia ", barr.nchunks)
