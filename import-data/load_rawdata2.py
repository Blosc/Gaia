# Script that loads the Gaia data and converts it to cartesian coordinates (in light years)

import glob
import os

import pandas as pd
import numpy as np
from time import time
import blosc2
import tables as tb

MAX_STARS = 1000_000_000
RADIUS = 10_000
LY_OFFSET = RADIUS
# First reduction factor
#R = 10
R = 1
CUBE_SIDE = 2 * RADIUS // R


def load_rawdata(outcoords=None, outtable=None):
    dtype = {"ra": np.float32, "dec": np.float32, "distance_gspphot": np.float32,
             "phot_g_mean_mag": np.float32, "radial_velocity": np.float32,
             "teff_gspphot": np.float32
             }
    dtype_table = {"idx": np.int64, "light_years": np.float32,
                   "phot_g_mean_mag": np.float32, "radial_velocity": np.float32,
                   "teff_gspphot": np.float32}
    h5file = tb.open_file(outtable, mode="w", title="Gaia data")
    table = h5file.create_table("/", "data", np.dtype(list(dtype_table.items())), "source data",
                                expectedrows=MAX_STARS,
                                filters=tb.Filters(complevel=1, complib="blosc2:zstd",
                                                   #least_significant_digit=2,
                                                   ))
    barr = None
    nstars = 0
    for i, file in enumerate(glob.glob("gaia-source/*.csv*")):
        print(f"{file=}")
        # Load raw data
        df = pd.read_csv(file, usecols=dtype.keys(), dtype=dtype, comment='#')
        #print(df.info())
        # Convert to numpy array and remove NaNs
        df = df.query(f"distance_gspphot > 0 and distance_gspphot < {RADIUS / 3.260}")
        # print(df.info())
        ra = df["ra"].to_numpy() * (np.pi / 180)
        dec = df["dec"].to_numpy() * (np.pi / 180)
        ly = df["distance_gspphot"].to_numpy() * 3.260  # convert to light years
        phot_g_mean_mag = df["phot_g_mean_mag"].to_numpy()
        radial_velocity = df["radial_velocity"].to_numpy()
        teff_gspphot = df["teff_gspphot"].to_numpy()
        # Compute x, y, z
        # https://www.jameswatkins.me/posts/converting-equatorial-to-cartesian.html
        x = (np.floor(ly * np.cos(dec) * np.cos(ra)) + LY_OFFSET) // R
        y = (np.floor(ly * np.cos(dec) * np.sin(ra)) + LY_OFFSET) // R
        z = (np.floor(ly * np.sin(dec)) + LY_OFFSET) // R
        if (x > CUBE_SIDE).any() or (y > CUBE_SIDE).any() or (z > CUBE_SIDE).any():
            print("WARNING: some stars are outside the cube")
            return None
        # Store in the array
        arr = np.vstack((x, y, z)).astype(np.int64)
        print("arr.shape=", arr.shape)
        if barr is None:
            barr = blosc2.asarray(arr, chunks=(3, 2**20), urlpath=outcoords, mode="w")
        else:
            barr.resize((3, barr.shape[1] + arr.shape[1]))
            barr[:, -arr.shape[1]:] = arr
        # Store in the Table object
        idx = x.astype(np.int64) * CUBE_SIDE * CUBE_SIDE + y.astype(np.int64) * CUBE_SIDE + z.astype(np.int64)
        rows = pd.DataFrame({"idx": idx, "light_years": ly,
                             "phot_g_mean_mag": phot_g_mean_mag,
                             "radial_velocity": radial_velocity,
                             "teff_gspphot": teff_gspphot}
                            ).to_records(index=False)
        table.append(rows)
        table.flush()
        nstars += arr.shape[1]
        if nstars > MAX_STARS:
            break
    h5file.close()
    return None


if __name__ == "__main__":
    rawdata = "gaia-ly-raw.b2nd"
    table_rawdata = "gaia-ly-table-raw.h5"
    t0 = time()
    load_rawdata(rawdata, table_rawdata)
    print(f"Time to load raw data: {time() - t0:.2f} s")

    # Copy on-disk to defragment and optimize for decompression
    t0 = time()
    arr = blosc2.open(rawdata)
    arr2 = blosc2.copy(arr, urlpath="gaia-ly.b2nd", mode="w",
                       # in case we want to truncate precision (not a big win anyway)
                       # cparams={
                       #     'filters': [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
                       #     'filters_meta': [16, 0],
                       #     }
                      )
    os.remove(rawdata)
    print(f"Time to copy and optimize: {time() - t0:.2f} s")
    print(f"Shape of raw data: {arr.shape=}, {arr.chunks=}, {arr.blocks=}")
    print(f"Size of raw data: {arr.schunk.nbytes / 2**30:.2f} GB ({arr.schunk.cratio:.2f}x)")

    # Do the same for the table
    t0 = time()
    fin = tb.open_file(table_rawdata, mode="r")
    table = fin.root.data
    fout = tb.open_file("gaia-ly-table.h5", mode="w")
    table.copy(fout.root, table.name, filters=tb.Filters(
        complevel=5, complib="blosc2:zstd"))
    fout.close()
    fin.close()
    os.remove(table_rawdata)
    print(f"Time to copy and optimize table: {time() - t0:.2f} s")
