from time import time

import numpy as np
import blosc2

R = 10
CUBE_SIDE = 20_000 // R
LY_OFFSET = 10_000
MAX_STARS = 1000_000

b = blosc2.open("gaia-ly.b2nd")
x = b[0, :MAX_STARS]
y = b[1, :MAX_STARS]
z = b[2, :MAX_STARS]

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(x, y, z, s=0.1)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()

t0 = time()
a3d = np.zeros((CUBE_SIDE, CUBE_SIDE, CUBE_SIDE), dtype=np.float32)
for x_, y_, z_ in zip(x, y, z):
    a3d[(int(x_) + LY_OFFSET) // R, (int(y_) + LY_OFFSET) // R, (int(z_) + LY_OFFSET) // R] += 1
print(f"Time to create 3d array: {time() - t0:.2f} s")

t0 = time()
blosc2.asarray(a3d, urlpath="gaia-3d.b2nd", mode="w",
               chunks=(200, 200, 200), blocks=(20, 20, 20),
               cparams={"clevel": 1, "codec": blosc2.Codec.ZSTD})

print(f"Time to store 3d array: {time() - t0:.2f} s")
