import blosc2
import numpy as np
import time
import sys


path_blosclz = sys.argv[1]
arr = blosc2.open(path_blosclz)
print(arr.info)


def bench(arr, start, stop, step, axis):
    t0 = time.time()
    while stop[axis] < arr.shape[axis]:
        _ = arr[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
        start[axis] += step
        stop[axis] += step
    return time.time() - t0


print("Axis = 0")
axis = 0
x_origin = 0
y_origin = 5_000
z_origin = 5_000
cube_side = 100
cube_shape = [cube_side, cube_side, cube_side]
start = np.array([x_origin, y_origin, z_origin])
stop = start + cube_shape[axis]
step = 25
t = bench(arr, start, stop, step, axis)
print(f"  *** NDArray.__getitem__ *** Time: {t:.3f} s")
print(f"  *** NDArray.__getitem__ *** Speed: {((arr.shape[0] / 25) * cube_side**3) / (2**30 * t):.3f} GB/s")

print("Axis = 1")
axis = 1
x_origin = 5_000
y_origin = 0
z_origin = 5_000
start = np.array([x_origin, y_origin, z_origin])
stop = start + cube_shape[axis]
t = bench(arr, start, stop, step, axis)
print(f"  *** NDArray.__getitem__ *** Time: {t:.3f} s")
print(f"  *** NDArray.__getitem__ *** Speed: {((arr.shape[0] / 25) * cube_side**3) / (2**30 * t):.3f} GB/s")

print("Axis = 2")
axis = 2
x_origin = 5_000
y_origin = 5_000
z_origin = 0
start = np.array([x_origin, y_origin, z_origin])
stop = start + cube_shape[axis]
t = bench(arr, start, stop, step, axis)
print(f"  *** NDArray.__getitem__ *** Time: {t:.3f} s")
print(f"  *** NDArray.__getitem__ *** Speed: {((arr.shape[0] / 25) * cube_side**3) / (2**30 * t):.3f} GB/s")

