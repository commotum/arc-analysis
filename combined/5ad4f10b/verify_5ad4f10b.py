import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_5ad4f10b(I: Grid) -> Grid:
    x0 = asindices(I)
    x1 = box(x0)
    x2 = toobject(x1, I)
    x3 = mostcolor(x2)
    x4 = palette(I)
    x5 = remove(x3, x4)
    x6 = lbind(chain, size)
    x7 = rbind(x6, dneighbors)
    x8 = lbind(lbind, intersection)
    x9 = lbind(ofcolor, I)
    x10 = chain(x7, x8, x9)
    x11 = rbind(matcher, ZERO)
    x12 = compose(x11, x10)
    x13 = chain(flip, positive, size)
    x14 = lbind(ofcolor, I)
    x15 = fork(sfilter, x14, x12)
    x16 = compose(x13, x15)
    x17 = argmax(x5, x16)
    x18 = other(x5, x17)
    x19 = ofcolor(I, x17)
    x20 = subgrid(x19, I)
    x21 = switch(x20, x17, x18)
    x22 = replace(x21, x17, x3)
    x23 = lbind(downscale, x22)
    x24 = fork(upscale, x23, identity)
    x25 = matcher(x24, x22)
    x26 = shape(x22)
    x27 = maximum(x26)
    x28 = interval(ONE, x27, ONE)
    x29 = sfilter(x28, x25)
    x30 = maximum(x29)
    x31 = downscale(x22, x30)
    return x31