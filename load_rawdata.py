import glob

import pandas as pd
import numpy as np
from time import time
import numexpr as ne
import blosc2
import os


def load_rawdata(out=None):
    dtype = {"ra": np.float32, "dec": np.float32, "parallax": np.float32, "phot_bp_mean_mag": np.float32}
    barr = None
    chunks = (2**20, 4)
    for i, file in enumerate(glob.glob("gaia-source/*.csv*")):
        # if i >= 10: break
        print(f"{file=}")
        # Load raw data
        df = pd.read_csv(file, usecols=["ra", "dec", "parallax", "phot_bp_mean_mag"], dtype=dtype, comment='#')
        # print(df.info())
        # Convert to numpy array and remove NaNs
        arr = df.to_numpy()
        arr = arr[~np.isnan(arr[:, 2])]
        if barr is None:
            # Store on a temporary file
            barr = blosc2.asarray(arr, chunks=chunks, urlpath=out, mode="w",
                                  cparams={
                #"clevel": 1, "codec": blosc2.Codec.ZSTD,
                # "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE],
                # "filters_meta": [0, 0],  # keep just 3 bits in mantissa
            })
        else:
            barr.resize((barr.shape[0] + arr.shape[0], 4))
            barr[-arr.shape[0]:] = arr

    return barr


def to_ly(fin, fout):
    barr = blosc2.open(fin)
    t0 = time()
    ra = barr[:, 0] * (np.pi / 180)
    dec = barr[:, 1] * (np.pi / 180)
    parallax = barr[:, 2]
    print(f"Time load data: {time() - t0:.2f} s")
    t0 = time()
    ly = ne.evaluate("3260 / parallax")  # approx. 1/parallax in light years
    # Remove ly < 0 and > 10000
    valid_ly = ne.evaluate("(ly > 0) & (ly < 10000)")
    print(f"valid_ly={valid_ly.sum()} ({valid_ly.sum() / len(ra):.2f}x)")
    ra = ra[valid_ly]
    dec = dec[valid_ly]
    ly = ly[valid_ly]
    print(f"Time to cut data for lightyears: {time() - t0:.2f} s")
    t0 = time()
    # Compute x, y, z
    # https://www.jameswatkins.me/posts/converting-equatorial-to-cartesian.html
    x = ne.evaluate("ly * cos(ra) * cos(dec)")
    y = ne.evaluate("ly * sin(ra) * cos(dec)")
    z = ne.evaluate("ly * sin(dec)")
    del ra, dec, parallax, ly
    print(f"Time to convert coordinates: {time() - t0:.2f} s")
    t0 = time()
    #g = barr[:, 3]
    #g = g[valid_ly]
    #del valid_ly
    #new_arr = np.stack((x, y, z, g), axis=1)
    #del x, y, z, g
    print(f"Time stack data: {time() - t0:.2f} s")
    t0 = time()
    out = blosc2.zeros(mode="w", shape=(4, len(x)), dtype=x.dtype, urlpath=fout)
    out[0, :] = x
    del x
    print(f"Time save x: {time() - t0:.2f} s")
    out[1, :] = y
    del y
    print(f"Time save y: {time() - t0:.2f} s")
    out[2, :] = z
    del z
    print(f"Time save z: {time() - t0:.2f} s")
    t1 = time()
    g = barr[:, 3]
    print(f"Time read g: {time() - t1:.2f} s")
    g = g[valid_ly]
    del valid_ly
    print(f"Time select g: {time() - t1:.2f} s")
    out[3, :] = g
    del g
    print(f"Time store g: {time() - t1:.2f} s")
    print(f"Time to store data: {time() - t0:.2f} s")

    return out


if __name__ == "__main__":
    tmp = "tmp.b2nd"
    outname = "gaia-ly.b2nd"
    # t0 = time()
    # arr = load_rawdata(tmp)
    # print(f"Time to load raw data: {time() - t0:.2f} s")
    # t0 = time()
    # arr = to_ly(tmp, outname)
    # # We don't need the original file anymore
    # os.remove(tmp)
    # print(f"Time to convert to light years: {time() - t0:.2f} s")

    # Copy on-disk to defragment and optimize for decompression
    os.rename(outname, tmp)
    arr2 = blosc2.open(tmp)
    arr = blosc2.copy(arr2, urlpath=outname, mode="w", contiguous=True,
                      cparams={
                          'clevel': 9, 'codec': blosc2.Codec.LZ4HC,
                          'filters': [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
                          'filters_meta': [16, 0],
                          }
                      )

    print(f"Shape of raw data: {arr.shape=}, {arr.chunks=}, {arr.blocks=}")
    print(f"Size of raw data: {arr.schunk.nbytes / 2**30:.2f} GB ({arr.schunk.cratio:.2f}x)")
    print(f"Number of chunks: {arr.schunk.nchunks}")
