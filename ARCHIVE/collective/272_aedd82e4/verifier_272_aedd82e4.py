import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_aedd82e4(I: Grid) -> Grid:
    x0 = shape(I)
    x1 = canvas(ZERO, x0)
    x2 = hconcat(I, x1)
    x3 = objects(x2, F, F, T)
    x4 = matcher(color, ZERO)
    x5 = compose(flip, x4)
    x6 = sfilter(x3, x5)
    x7 = sizefilter(x6, ONE)
    x8 = merge(x7)
    x9 = fill(I, ONE, x8)
    return x9