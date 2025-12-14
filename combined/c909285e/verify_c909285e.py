import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_c909285e(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = lbind(contained, ONE)
    x2 = chain(flip, x1, shape)
    x3 = sfilter(x0, x2)
    x4 = fork(equality, toindices, box)
    x5 = sfilter(x3, x4)
    x6 = fork(multiply, height, width)
    x7 = argmin(x5, x6)
    x8 = subgrid(x7, I)
    return x8