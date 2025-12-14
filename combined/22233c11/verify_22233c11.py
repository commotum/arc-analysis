import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_22233c11(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = rbind(upscale, TWO)
    x2 = chain(invert, halve, shape)
    x3 = fork(combine, hfrontier, vfrontier)
    x4 = compose(x1, vmirror)
    x5 = fork(shift, x4, x2)
    x6 = compose(toindices, x5)
    x7 = lbind(mapply, x3)
    x8 = compose(x7, toindices)
    x9 = fork(difference, x6, x8)
    x10 = mapply(x9, x0)
    x11 = fill(I, EIGHT, x10)
    return x11