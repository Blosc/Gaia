import htc
from pathlib import Path
import blosc2

f = htc.decompress_file(Path("/home/marta/BTune-training/analysis/P093#2021_04_28_08_49_12.blosc"))

print(f)
print(f.shape)
print(f.dtype)

array = blosc2.asarray(f, chunks=(64, 64, 64), urlpath="../data/cancers2.b2nd", mode="w")
print(array.info)
