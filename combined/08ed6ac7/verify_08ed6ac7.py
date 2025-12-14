import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_08ed6ac7(I: Grid) -> Grid:
    x0 = first(I)
    x1 = mostcommon(x0)
    x2 = dmirror(I)
    x3 = matcher(identity, x1)
    x4 = rbind(sfilter, x3)
    x5 = compose(size, x4)
    x6 = apply(x5, x2)
    x7 = dedupe(x6)
    x8 = order(x7, identity)
    x9 = size(x8)
    x10 = increment(x9)
    x11 = increment(x10)
    x12 = interval(ONE, x11, ONE)
    x13 = pair(x8, x12)
    x14 = height(I)
    x15 = astuple(x14, x1)
    x16 = repeat(x15, ONE)
    x17 = combine(x16, x13)
    x18 = lbind(extract, x17)
    x19 = lbind(matcher, first)
    x20 = chain(last, x18, x19)
    x21 = compose(x20, x5)
    x22 = fork(subtract, height, x5)
    x23 = fork(repeat, x21, x22)
    x24 = lbind(repeat, x1)
    x25 = compose(x24, x5)
    x26 = fork(combine, x25, x23)
    x27 = apply(x26, x2)
    x28 = dmirror(x27)
    return x28