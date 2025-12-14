import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_7e0986d6(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = objects(I, T, F, T)
    x2 = lbind(greater, THREE)
    x3 = compose(x2, size)
    x4 = sfilter(x1, x3)
    x5 = mapply(toindices, x4)
    x6 = fill(I, x0, x5)
    x7 = objects(x6, T, F, T)
    x8 = fork(recolor, color, backdrop)
    x9 = mapply(x8, x7)
    x10 = paint(x6, x9)
    return x10