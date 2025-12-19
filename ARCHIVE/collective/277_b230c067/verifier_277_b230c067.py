import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_b230c067(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = lbind(sfilter, x0)
    x2 = lbind(matcher, normalize)
    x3 = compose(x2, normalize)
    x4 = chain(size, x1, x3)
    x5 = argmin(x0, x4)
    x6 = remove(x5, x0)
    x7 = merge(x6)
    x8 = fill(I, TWO, x5)
    x9 = fill(x8, ONE, x7)
    return x9