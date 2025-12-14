import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_c9f8e694(I: Grid) -> Grid:
    x0 = astuple(identity, dmirror)
    x1 = astuple(cmirror, vmirror)
    x2 = combine(x0, x1)
    x3 = compose(first, dmirror)
    x4 = chain(size, dedupe, x3)
    x5 = rbind(rapply, I)
    x6 = compose(first, x5)
    x7 = chain(x4, x6, initset)
    x8 = argmax(x2, x7)
    x9 = x8(I)
    x10 = height(x9)
    x11 = width(x9)
    x12 = ofcolor(x9, ZERO)
    x13 = astuple(x10, ONE)
    x14 = crop(x9, ORIGIN, x13)
    x15 = hupscale(x14, x11)
    x16 = fill(x15, ZERO, x12)
    x17 = x8(x16)
    return x17