import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_6e82a1ae(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = matcher(size, TWO)
    x2 = mfilter(x0, x1)
    x3 = matcher(size, THREE)
    x4 = mfilter(x0, x3)
    x5 = matcher(size, FOUR)
    x6 = mfilter(x0, x5)
    x7 = fill(I, THREE, x2)
    x8 = fill(x7, TWO, x4)
    x9 = fill(x8, ONE, x6)
    return x9