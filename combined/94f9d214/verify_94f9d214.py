import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_94f9d214(I: Grid) -> Grid:
    x0 = astuple(vsplit, hsplit)
    x1 = rbind(rbind, TWO)
    x2 = rbind(rapply, I)
    x3 = initset(x1)
    x4 = lbind(rapply, x3)
    x5 = chain(first, x2, x4)
    x6 = lbind(apply, numcolors)
    x7 = compose(x6, x5)
    x8 = matcher(x7, TWO_BY_TWO)
    x9 = extract(x0, x8)
    x10 = x9(I, TWO)
    x11 = first(x10)
    x12 = last(x10)
    x13 = palette(x11)
    x14 = palette(x12)
    x15 = intersection(x13, x14)
    x16 = first(x15)
    x17 = shape(x11)
    x18 = canvas(x16, x17)
    x19 = ofcolor(x11, x16)
    x20 = ofcolor(x12, x16)
    x21 = intersection(x19, x20)
    x22 = fill(x18, TWO, x21)
    return x22