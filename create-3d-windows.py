# Create a 3d array with size RADIUS x RADIUS x RADIUS, by using windows of size
# RADIUS/R2 x RADIUS/R2 x RADIUS/R2 which fits in memory.

from time import time
import math

import numpy as np
import numexpr as ne
import blosc2

MAX_STARS = 10_000_000_000
RADIUS = 20_000
LY_OFFSET = RADIUS // 2
# First reduction factor
#R = 10
R = 1
CUBE_SIDE = RADIUS // R

t0 = time()
b = blosc2.open("gaia-ly-3.b2nd")
x = b[0, :MAX_STARS]
y = b[1, :MAX_STARS]
z = b[2, :MAX_STARS]
print(f"Time to read: {time() - t0:.2f} s")


def create_windows(x, y, z, RADIUS, LY_OFFSET, R2, dtype=np.uint8):
    shape = (CUBE_SIDE, CUBE_SIDE, CUBE_SIDE)
    b3d = blosc2.zeros(shape, dtype=dtype, urlpath="gaia-3d-windows-int8-3.b2nd", mode="w",
                       chunks=(250, 250, 250), blocks=(25, 25, 25),
                       cparams={"clevel": 1, "codec": blosc2.Codec.ZSTD})
    CUBE_SIDE2 = CUBE_SIDE // R2
    RADIUS2 = RADIUS // R2
    total_stars = 0
    nstars = 0
    t0 = time()
    t1 = time()
    for i in range(R2):
        _x = ne.evaluate(
            "((x + LY_OFFSET) >= i * RADIUS2) & ((x + LY_OFFSET) < (i + 1) * RADIUS2)")
        x_offset = LY_OFFSET - i * RADIUS2
        for j in range(R2):
            _y = ne.evaluate(
                "((y + LY_OFFSET) >= j * RADIUS2) & ((y + LY_OFFSET) < (j + 1) * RADIUS2)")
            y_offset = LY_OFFSET - j * RADIUS2
            for k in range(R2):
                print(f"{i=}, {j=}, {k=} t0: {time() - t0:.4f} s, t1: {time() - t1:.4f} s")
                t1 = time()
                _z = ne.evaluate(
                    "((z + LY_OFFSET) >= k * RADIUS2) & ((z + LY_OFFSET) < (k + 1) * RADIUS2)")
                z_offset = LY_OFFSET - k * RADIUS2
                stars_in_window = ne.evaluate("_x & _y & _z")
                nstars_in_window = np.sum(stars_in_window)
                if nstars_in_window == 0:
                    continue
                total_stars += nstars_in_window
                print("Number of stars in window and total:", nstars_in_window, total_stars)
                star_x = x[stars_in_window]
                star_y = y[stars_in_window]
                star_z = z[stars_in_window]
                a3d = np.zeros((CUBE_SIDE2, CUBE_SIDE2, CUBE_SIDE2), dtype=dtype)
                for x_, y_, z_ in zip(star_x, star_y, star_z):
                    x2 = (math.floor(x_) + x_offset) // R
                    y2 = (math.floor(y_) + y_offset) // R
                    z2 = (math.floor(z_) + z_offset) // R
                    a3d[x2, y2, z2] += 1
                nstars += np.sum(a3d)
                slice_ = (slice(i * CUBE_SIDE2, (i + 1) * CUBE_SIDE2),
                          slice(j * CUBE_SIDE2, (j + 1) * CUBE_SIDE2),
                          slice(k * CUBE_SIDE2, (k + 1) * CUBE_SIDE2))
                b3d[slice_] = a3d
    print(f"{nstars=} {total_stars=}")
    return b3d

# Windows with another R2 reduction factor
t0 = time()
b3dbis = create_windows(x, y, z, RADIUS, LY_OFFSET, R2=10, dtype=np.uint8)
print(f"Time to create outer 3d array: {time() - t0:.2f} s")
#print(b3d.info)