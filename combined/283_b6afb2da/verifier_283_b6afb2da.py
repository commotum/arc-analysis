import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b6afb2da(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = fork(equality, toindices, backdrop)
    x2 = compose(flip, x1)
    x3 = extract(x0, x2)
    x4 = color(x3)
    x5 = matcher(color, x4)
    x6 = compose(flip, x5)
    x7 = sfilter(x0, x6)
    x8 = merge(x7)
    x9 = fill(I, TWO, x8)
    x10 = mapply(box, x7)
    x11 = fill(x9, FOUR, x10)
    x12 = mapply(corners, x7)
    x13 = fill(x11, ONE, x12)
    return x13