import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_67385a82(I: Grid) -> Grid:
    x0 = objects(I, T, F, F)
    x1 = palette(I)
    x2 = other(x1, ZERO)
    x3 = colorfilter(x0, x2)
    x4 = sizefilter(x3, ONE)
    x5 = difference(x3, x4)
    x6 = merge(x5)
    x7 = fill(I, EIGHT, x6)
    return x7