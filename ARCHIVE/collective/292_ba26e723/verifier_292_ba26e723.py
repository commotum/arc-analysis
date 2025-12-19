import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ba26e723(I: Grid) -> Grid:
    x0 = asobject(I)
    x1 = matcher(first, ZERO)
    x2 = compose(flip, x1)
    x3 = sfilter(x0, x2)
    x4 = rbind(multiply, THREE)
    x5 = rbind(divide, THREE)
    x6 = compose(x4, x5)
    x7 = fork(equality, identity, x6)
    x8 = toindices(x3)
    x9 = compose(x7, last)
    x10 = sfilter(x8, x9)
    x11 = fill(I, SIX, x10)
    return x11