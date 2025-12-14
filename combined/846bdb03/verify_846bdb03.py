import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_846bdb03(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(equality, corners, toindices)
    x2 = extract(x0, x1)
    x3 = subgrid(x2, I)
    x4 = backdrop(x2)
    x5 = cover(I, x4)
    x6 = frontiers(x3)
    x7 = sfilter(x6, hline)
    x8 = size(x7)
    x9 = positive(x8)
    x10 = branch(x9, dmirror, identity)
    x11 = x10(x3)
    x12 = x10(x5)
    x13 = fgpartition(x12)
    x14 = merge(x13)
    x15 = normalize(x14)
    x16 = mostcolor(x12)
    x17 = color(x2)
    x18 = palette(x11)
    x19 = remove(x17, x18)
    x20 = remove(x16, x19)
    x21 = first(x20)
    x22 = last(x20)
    x23 = ofcolor(x11, x22)
    x24 = leftmost(x23)
    x25 = ofcolor(x11, x21)
    x26 = leftmost(x25)
    x27 = greater(x24, x26)
    x28 = ofcolor(x12, x22)
    x29 = leftmost(x28)
    x30 = ofcolor(x12, x21)
    x31 = leftmost(x30)
    x32 = greater(x29, x31)
    x33 = equality(x27, x32)
    x34 = branch(x33, identity, vmirror)
    x35 = x34(x15)
    x36 = shift(x35, UNITY)
    x37 = paint(x11, x36)
    x38 = x10(x37)
    return x38