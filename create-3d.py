from time import time
import numpy as np
import blosc2

R = 2
CUBE_SIDE = 20_000 // R
LY_OFFSET = CUBE_SIDE // 2
MAX_STARS = 1_000_000_000

b = blosc2.open("gaia-ly.b2nd")
x = b[0, :MAX_STARS]
y = b[1, :MAX_STARS]
z = b[2, :MAX_STARS]
g = b[3, :MAX_STARS]

print("len coords:", len(g))
t0 = time()
a3d = np.zeros((CUBE_SIDE, CUBE_SIDE, CUBE_SIDE), dtype=np.float32)
for i, coords in enumerate(zip(x, y, z)):
    x_, y_, z_ = coords
    a3d[(int(x_) + LY_OFFSET) // R, (int(y_) + LY_OFFSET) // R, (int(z_) + LY_OFFSET) // R] += g[i]
print(f"Time to create 3d array: {time() - t0:.2f} s")

t0 = time()
blosc2.asarray(a3d, urlpath="gaia-3d.b2nd", mode="w",
               chunks=(200, 200, 200), blocks=(20, 20, 20),
               cparams={"clevel": 1, "codec": blosc2.Codec.ZSTD})

print(f"Time to store 3d array: {time() - t0:.2f} s")
