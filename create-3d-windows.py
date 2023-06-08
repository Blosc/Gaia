from time import time

import numpy as np
import numexpr as ne
import blosc2

R = 10
RADIUS = 20_000
MAX_STARS = 10_000
CUBE_SIDE = RADIUS // R
LY_OFFSET = RADIUS // 2

t0 = time()
b = blosc2.open("gaia-ly.b2nd")
x = b[0, :MAX_STARS]
y = b[1, :MAX_STARS]
z = b[2, :MAX_STARS]
print(f"Time to read: {time() - t0:.2f} s")

t0 = time()
R2 = 10 * R
CUBE_SIDE2 = RADIUS // R2
shape = (CUBE_SIDE, CUBE_SIDE, CUBE_SIDE)
b3d = blosc2.empty(shape, dtype=np.float32, # urlpath="gaia-3d-windows.b2nd", mode="w",
                   chunks=(200, 200, 200), blocks=(20, 20, 20),
                   cparams={"clevel": 1, "codec": blosc2.Codec.ZSTD})
total_stars = 0
for i in range(R2):
    _x = ne.evaluate("((x + LY_OFFSET) >= i * CUBE_SIDE2) & ((x + LY_OFFSET) < (i + 1) * CUBE_SIDE2)")
    x_offset = LY_OFFSET - i * CUBE_SIDE2
    for j in range(R2):
        _y = ne.evaluate("((y + LY_OFFSET) >= j * CUBE_SIDE2) & ((y + LY_OFFSET) < (j + 1) * CUBE_SIDE2)")
        y_offset = LY_OFFSET - j * CUBE_SIDE2
        for k in range(R2):
            #t1 = time()
            _z = ne.evaluate("((z + LY_OFFSET) >= k * CUBE_SIDE2) & ((z + LY_OFFSET) < (k + 1) * CUBE_SIDE2)")
            #print(f"Time to evaluate: {time() - t1:.2f} s")
            stars_in_window = ne.evaluate("_x & _y & _z")
            nstars_in_window = np.sum(stars_in_window)
            if nstars_in_window == 0:
                continue
            total_stars += nstars_in_window
            print(f"{i=}, {j=}, {k=}")
            print("Number of stars in window:", nstars_in_window)
            star_x = x[stars_in_window]
            star_y = y[stars_in_window]
            star_z = z[stars_in_window]
            z_offset = LY_OFFSET - k * CUBE_SIDE2
            a3d = np.zeros((CUBE_SIDE2, CUBE_SIDE2, CUBE_SIDE2), dtype=np.float32)
            t1 = time()
            for x_, y_, z_ in zip(star_x, star_y, star_z):
                x2 = (int(x_) + x_offset) // R2
                y2 = (int(y_) + y_offset) // R2
                z2 = (int(z_) + z_offset) // R2
                a3d[x2, y2, z2] += 1
            print(f"Time to create inner 3d array: {time() - t1:.3f} s")
            b3d[i * CUBE_SIDE2:(i + 1) * CUBE_SIDE2,
                j * CUBE_SIDE2:(j + 1) * CUBE_SIDE2,
                k * CUBE_SIDE2:(k + 1) * CUBE_SIDE2] = a3d

print(f"Total number of stars: {total_stars}")
print(f"Time to create outer 3d array: {time() - t0:.2f} s")
print(b3d.info)
