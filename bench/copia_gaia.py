import blosc2
import numpy as np
from time import time


# path = "/home/blosc/gaia/gaia-3d-windows.b2nd"
path = "copiagaia-blosclz5.b2nd"
arr = blosc2.open(path)


cparams = {}
#cparams["codec"] = blosc2.Codec.BLOSCLZ
#cparams["codec"] = blosc2.Codec.LZ4
cparams["codec"] = blosc2.Codec.ZSTD
cparams["filters"] = [blosc2.Filter.NOFILTER] * 6
#cparams["filters"][0] = blosc2.Filter.BITSHUFFLE
cparams["filters_meta"] = [0] * 6
cparams["splitmode"] = blosc2.SplitMode.NEVER_SPLIT
cparams["clevel"] = 1
#cparams["clevel"] = 5
#cparams["clevel"] = 9
print("cparams ", cparams)
storage = {"cparams": cparams, "urlpath": "copiagaia-zstd1.b2nd"}
blosc2.remove_urlpath(storage["urlpath"])

t0 = time()
barr = arr.copy(chunks=arr.chunks, blocks=arr.blocks, **storage)
t = time() - t0
print("Time in seconds for copying array ", t)
print(arr.info)
print(barr.info)

