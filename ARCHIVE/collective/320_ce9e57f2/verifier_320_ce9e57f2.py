import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ce9e57f2(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = asindices(I)
    x2 = outbox(x1)
    x3 = lbind(adjacent, x2)
    x4 = compose(x3, initset)
    x5 = rbind(extract, x4)
    x6 = compose(x5, toindices)
    x7 = rbind(compose, initset)
    x8 = lbind(rbind, manhattan)
    x9 = chain(x7, x8, initset)
    x10 = lbind(lbind, greater)
    x11 = chain(x10, halve, size)
    x12 = compose(x9, x6)
    x13 = fork(compose, x11, x12)
    x14 = fork(sfilter, toindices, x13)
    x15 = mapply(x14, x0)
    x16 = fill(I, EIGHT, x15)
    return x16