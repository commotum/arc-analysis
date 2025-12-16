import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d2abd087(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = matcher(size, SIX)
    x2 = compose(flip, x1)
    x3 = mfilter(x0, x1)
    x4 = mfilter(x0, x2)
    x5 = fill(I, TWO, x3)
    x6 = fill(x5, ONE, x4)
    return x6