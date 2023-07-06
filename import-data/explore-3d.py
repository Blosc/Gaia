# Script to create a 3d array from the Gaia dataset.  This needs to keep the final 3d NumPy array
# in memory, so it can only be used for such cubes that fit in (virtual) memory.

from time import time
import blosc2

R = 10
CUBE_SIDE = 20_000 // R
LY_OFFSET = CUBE_SIDE // 2

t0 = time()
b3d = blosc2.open(urlpath="gaia-3d.b2nd", mode="r")
print(f"Time to open 3d array: {time() - t0:.2f} s")
for i in range(10):
    low, high = b3d.schunk.blocks, LY_OFFSET + 200
    a3d = b3d[low:high, low:high, low:high]

low, high = LY_OFFSET - 200, LY_OFFSET + 200
#a3d = b3d[low:high, low:high, low:high]
a3d = b3d[...]

print(a3d.sum(), a3d.max())

