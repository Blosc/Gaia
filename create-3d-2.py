# Script to create a 3d array from the Gaia dataset.  This needs to keep the final 3d NumPy array
# in memory, so it can only be used for such cubes that fit in (virtual) memory.

from time import time
import numpy as np
import blosc2
import math

RADIUS = 10_000
R = 10
CUBE_SIDE = 2 * RADIUS // R
MAX_STARS = 1_000_000_000

b = blosc2.open("gaia-ly.b2nd")
x = b[0, :MAX_STARS]
y = b[1, :MAX_STARS]
z = b[2, :MAX_STARS]

print("len coords:", len(x))
t0 = time()
a3d = np.zeros((CUBE_SIDE, CUBE_SIDE, CUBE_SIDE), dtype=np.uint8)
for coords in zip(x, y, z):
    x_, y_, z_ = coords
    a3d[math.floor(x_ / R), math.floor(y_ / R), math.floor(z_ / R)] += 1
print(f"Time to create 3d array: {time() - t0:.2f} s")
# print("Number of stars in 3d array:", np.sum(a3d))
# print("Number of non-zero elements in 3d array:", np.count_nonzero(a3d))
# print("Max value in cells:", np.max(a3d))

t0 = time()
blosc2.asarray(a3d, urlpath="gaia-3d.b2nd", mode="w",
               chunks=(250, 250, 250), blocks=(25, 25, 25),
               cparams={"clevel": 5, "codec": blosc2.Codec.ZSTD})

print(f"Time to store 3d array: {time() - t0:.2f} s")
