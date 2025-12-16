import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b2862040(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = mostcolor(I)
    x2 = colorfilter(x0, x1)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    x6 = difference(x0, x2)
    x7 = apply(toindices, x6)
    x8 = rbind(adjacent, x5)
    x9 = mfilter(x7, x8)
    x10 = fill(I, EIGHT, x9)
    return x10