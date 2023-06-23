# Script that loads the Gaia data and converts it to cartesian coordinates (in light years)

import glob

import pandas as pd
import numpy as np
from time import time
import numexpr as ne
import blosc2
import os


def load_rawdata(out=None):
    dtype = {"ra": np.float32, "dec": np.float32, "parallax": np.float32}
    barr = None
    chunks = (2**20, 3)
    for i, file in enumerate(glob.glob("gaia-source/*.csv*")):
        print(f"{file=}")
        # Load raw data
        df = pd.read_csv(file, usecols=["ra", "dec", "parallax"], dtype=dtype, comment='#')
        # print(df.info())
        # Convert to numpy array and remove NaNs
        arr = df.to_numpy()
        arr = arr[~np.isnan(arr[:, 2])]
        if barr is None:
            # Store on a temporary file
            barr = blosc2.asarray(arr, chunks=chunks, urlpath=out, mode="w")
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
    ly = ne.evaluate("3260 / parallax")  # approx. 1/parallax in light years
    # Remove ly < 0 and > 10000
    valid_ly = ne.evaluate("(ly > 0) & (ly < 10000)")
    print(f"valid_ly={valid_ly.sum()} ({valid_ly.sum() / len(ra):.2f}x)")
    ra = ra[valid_ly]
    dec = dec[valid_ly]
    ly = ly[valid_ly]
    # Compute x, y, z
    # https://www.jameswatkins.me/posts/converting-equatorial-to-cartesian.html
    x = ne.evaluate("ly * cos(ra) * cos(dec)")
    y = ne.evaluate("ly * sin(ra) * cos(dec)")
    z = ne.evaluate("ly * sin(dec)")
    del ra, dec, parallax, ly
    out = blosc2.zeros(mode="w", shape=(4, len(x)), dtype=x.dtype, urlpath=fout)
    out[0, :] = x
    del x
    out[1, :] = y
    del y
    out[2, :] = z
    del z
    del valid_ly

    return out


if __name__ == "__main__":
    rawname = "raw.b2nd"
    outname = "gaia-ly.b2nd"
    t0 = time()
    arr = load_rawdata(rawname)
    print(f"Time to load raw data: {time() - t0:.2f} s")
    t0 = time()
    arr = to_ly(rawname, outname)
    print(f"Time to convert to light years: {time() - t0:.2f} s")

    # Copy on-disk to defragment and optimize for decompression
    t0 = time()
    tmp = "tmp.b2nd"
    os.rename(outname, tmp)
    arr2 = blosc2.open(tmp)
    arr = blosc2.copy(arr2, urlpath=outname, mode="w", contiguous=True,
                      cparams={
                          'clevel': 9, 'codec': blosc2.Codec.LZ4HC,
                          'filters': [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
                          'filters_meta': [16, 0],
                          }
                      )
    os.remove(tmp)
    print(f"Time to copy and optimize: {time() - t0:.2f} s")
    print(f"Shape of raw data: {arr.shape=}, {arr.chunks=}, {arr.blocks=}")
    print(f"Size of raw data: {arr.schunk.nbytes / 2**30:.2f} GB ({arr.schunk.cratio:.2f}x)")
