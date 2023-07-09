import os
paths = ['copiagaia0.b2nd', 'copiagaia6.b2nd', 'copiagaia10.b2nd', 'copiagaiablock_auto.b2nd', 'zarrcopia', 'h5pycopy.hdf5', 'h5pycopy-blosc1.hdf5']

for path in paths:
    if 'zarr' not in path:
        print("size in GB of ", path, " : ", os.path.getsize(path) / 2**30)
    else:
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
        print("size in GB of ", path, " : ", total / 2**30)
