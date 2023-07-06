# Script that loads some of the Gaia parameters into a PyTables Table.  The table will be indexed
# by the Gaia 3d coordinates, so it can be used to retrieve the parameters of a given star.

import glob
import os

import pandas as pd
import numpy as np
from time import time
import tables as tb


MAX_STARS = 1000_000_000
RADIUS = 20_000
LY_OFFSET = RADIUS // 2
# First reduction factor
#R = 10
R = 1
CUBE_SIDE = RADIUS // R


def load_table(outname=None):
    t0 = time()
    dtype_csv = {"ra": np.float32, "dec": np.float32, "parallax": np.float32,
                 "phot_g_mean_mag": np.float32, "radial_velocity": np.float32,
                 "teff_gspphot": np.float32}
    dtype_table = {"idx": np.int64,
                   "phot_g_mean_mag": np.float32, "radial_velocity": np.float32,
                   "teff_gspphot": np.float32}
    h5file = tb.open_file(outname, mode="w", title="Gaia data")
    table = h5file.create_table("/", "data", np.dtype(list(dtype_table.items())), "source data",
                                expectedrows=MAX_STARS,
                                filters=tb.Filters(complevel=1, complib="blosc2:zstd",
                                                   #least_significant_digit=2,
                                                   ))
    nstars = 0
    for i, file in enumerate(glob.glob("gaia-source/*.csv*")):
        # if i >= 10: break
        print(f"{file=}")
        # Load raw data
        df = pd.read_csv(file, usecols=dtype_csv.keys(), dtype=dtype_csv, comment='#')
        #print(f"Time to load data: {time() - t0:.2f} s")
        # Convert to numpy array and remove NaNs
        arr = df.to_numpy()
        arr = arr[~np.isnan(arr[:, 2])]
        ra = arr[:, 0] * (np.pi / 180)
        dec = arr[:, 1] * (np.pi / 180)
        parallax = arr[:, 2]
        #print(f"Time to convert to NumPy and remove NaNs: {time() - t0:.2f} s")
        ly = 3260 / parallax  # approx. 1/parallax in light years
        # Remove ly < 0 and > 10000
        valid_ly = (ly > 0) & (ly < 10000)
        #print(f"valid_ly={valid_ly.sum()} ({valid_ly.sum() / len(ra):.2f}x)")
        ra = ra[valid_ly]
        dec = dec[valid_ly]
        ly = ly[valid_ly]
        phot_g_mean_mag = arr[:, 3][valid_ly]
        radial_velocity = arr[:, 4][valid_ly]
        teff_gspphot = arr[:, 5][valid_ly]
        #print(f"Time to get valid coordinates: {time() - t0:.2f} s")
        # Compute x, y, z
        # https://www.jameswatkins.me/posts/converting-equatorial-to-cartesian.html
        x = (np.floor(ly * np.cos(dec) * np.cos(ra)) + LY_OFFSET) // R
        y = (np.floor(ly * np.cos(dec) * np.sin(ra)) + LY_OFFSET) // R
        z = (np.floor(ly * np.sin(dec)) + LY_OFFSET) // R
        idx = x.astype(np.int64) * CUBE_SIDE * CUBE_SIDE + y.astype(np.int64) * CUBE_SIDE + z.astype(np.int64)
        #print(f"Time compute cartesian coords: {time() - t0:.2f} s")
        # Store in the Table object
        rows = pd.DataFrame({"idx": idx, "phot_g_mean_mag": phot_g_mean_mag,
                              "radial_velocity": radial_velocity, "teff_gspphot": teff_gspphot}
                            ).to_records(index=False)
        #print(f"Time to convert to pandas: {time() - t0:.2f} s")
        table.append(rows)
        table.flush()
        #print(f"Time to append to table: {time() - t0:.2f} s")
        nstars += len(idx)
        if nstars >= MAX_STARS:
            break

    return h5file


if __name__ == "__main__":
    outname = "gaia-table.h5"
    t0 = time()
    h5file = load_table(outname)
    table = h5file.root.data
    print(f"Time to load raw table: {time() - t0:.2f} s")
    print(f"Elements in table: {table.nrows=}")
    nbytes = table.nrows * table.dtype.itemsize
    fsize = os.path.getsize(outname)
    print(f"Size of table: {nbytes / 2**30:.2f} GB ({nbytes / fsize:.2f}x)")

    t0 = time()
    #table.cols.idx.create_index(optlevel=6, kind="medium")
    #table.cols.idx.create_index(optlevel=9, kind="full")
    table.cols.idx.create_csindex()
    table.flush()
    print(f"Time to create index: {time() - t0:.2f} s")
    fsize_idx = os.path.getsize(outname) - fsize
    print(f"Size of index: {fsize_idx / 2**20:.2f} MB")

    h5file.close()
