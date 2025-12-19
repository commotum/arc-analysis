import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1c786137(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = lbind(colorfilter, x0)
    x2 = compose(size, x1)
    x3 = matcher(x2, ONE)
    x4 = palette(I)
    x5 = sfilter(x4, x3)
    x6 = fork(equality, toindices, box)
    x7 = rbind(contained, x5)
    x8 = compose(x7, color)
    x9 = sfilter(x0, x8)
    x10 = rbind(greater, SEVEN)
    x11 = compose(x10, size)
    x12 = sfilter(x9, x11)
    x13 = extract(x12, x6)
    x14 = subgrid(x13, I)
    x15 = trim(x14)
    return x15