import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_776ffc46(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = fork(equality, toindices, box)
    x2 = sfilter(x0, x1)
    x3 = fork(multiply, height, width)
    x4 = argmax(x2, x3)
    x5 = mostcolor(I)
    x6 = inbox(x4)
    x7 = backdrop(x6)
    x8 = toobject(x7, I)
    x9 = matcher(first, x5)
    x10 = compose(flip, x9)
    x11 = sfilter(x8, x10)
    x12 = normalize(x11)
    x13 = color(x12)
    x14 = toindices(x12)
    x15 = compose(toindices, normalize)
    x16 = matcher(x15, x14)
    x17 = mfilter(x0, x16)
    x18 = fill(I, x13, x17)
    return x18