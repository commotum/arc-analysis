import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1f85a75f(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = totuple(x0)
    x2 = apply(color, x1)
    x3 = lbind(sfilter, x2)
    x4 = lbind(matcher, identity)
    x5 = chain(size, x3, x4)
    x6 = matcher(x5, ONE)
    x7 = sfilter(x2, x6)
    x8 = lbind(colorcount, I)
    x9 = argmax(x7, x8)
    x10 = matcher(color, x9)
    x11 = extract(x0, x10)
    x12 = subgrid(x11, I)
    return x12