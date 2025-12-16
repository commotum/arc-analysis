import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_3bd67248(I: Grid) -> Grid:
    x0 = astuple(identity, identity)
    x1 = astuple(rot90, rot270)
    x2 = astuple(x0, x1)
    x3 = astuple(rot180, rot180)
    x4 = astuple(rot270, rot90)
    x5 = astuple(x3, x4)
    x6 = combine(x2, x5)
    x7 = leastcolor(I)
    x8 = repeat(x7, ONE)
    x9 = rbind(rapply, I)
    x10 = chain(x9, initset, first)
    x11 = compose(first, x10)
    x12 = chain(dedupe, first, x11)
    x13 = matcher(x12, x8)
    x14 = extract(x6, x13)
    x15 = first(x14)
    x16 = last(x14)
    x17 = x15(I)
    x18 = ofcolor(x17, x7)
    x19 = height(x18)
    x20 = interval(ZERO, x19, ONE)
    x21 = lbind(astuple, x19)
    x22 = apply(x21, x20)
    x23 = rbind(shoot, DOWN)
    x24 = mapply(x23, x22)
    x25 = fill(x17, FOUR, x24)
    x26 = astuple(x19, x19)
    x27 = canvas(ZERO, x26)
    x28 = asindices(x27)
    x29 = shift(x28, x26)
    x30 = shape(I)
    x31 = maximum(x30)
    x32 = lbind(shift, x29)
    x33 = interval(ZERO, x31, x19)
    x34 = pair(x33, x33)
    x35 = mapply(x32, x34)
    x36 = fill(x25, TWO, x35)
    x37 = x16(x36)
    return x37