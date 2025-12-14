import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_2204b7a8(I: Grid) -> Grid:
    x0 = first(I)
    x1 = dedupe(x0)
    x2 = size(x1)
    x3 = equality(x2, ONE)
    x4 = flip(x3)
    x5 = branch(x4, lefthalf, tophalf)
    x6 = branch(x4, righthalf, bottomhalf)
    x7 = branch(x4, hconcat, vconcat)
    x8 = x5(I)
    x9 = x6(I)
    x10 = index(x8, ORIGIN)
    x11 = shape(x9)
    x12 = decrement(x11)
    x13 = index(x9, x12)
    x14 = mostcolor(I)
    x15 = mostcolor(I)
    x16 = palette(I)
    x17 = remove(x10, x16)
    x18 = remove(x13, x17)
    x19 = remove(x15, x18)
    x20 = first(x19)
    x21 = replace(x8, x20, x10)
    x22 = branch(x4, dmirror, identity)
    x23 = branch(x4, height, width)
    x24 = x23(I)
    x25 = astuple(ONE, x24)
    x26 = canvas(x14, x25)
    x27 = x22(x26)
    x28 = replace(x9, x20, x13)
    x29 = x7(x21, x27)
    x30 = branch(x4, width, height)
    x31 = x30(I)
    x32 = even(x31)
    x33 = branch(x32, x21, x29)
    x34 = x7(x33, x28)
    return x34