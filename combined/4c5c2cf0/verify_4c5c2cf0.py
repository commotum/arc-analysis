import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_4c5c2cf0(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = compose(dneighbors, center)
    x2 = fork(difference, backdrop, x1)
    x3 = fork(equality, toindices, x2)
    x4 = matcher(size, FIVE)
    x5 = fork(both, x3, x4)
    x6 = extract(x0, x5)
    x7 = color(x6)
    x8 = merge(x0)
    x9 = compose(hmirror, vmirror)
    x10 = initset(x9)
    x11 = insert(vmirror, x10)
    x12 = insert(hmirror, x11)
    x13 = rapply(x12, x8)
    x14 = ulcorner(x6)
    x15 = lbind(subtract, x14)
    x16 = matcher(first, x7)
    x17 = rbind(sfilter, x16)
    x18 = chain(x15, ulcorner, x17)
    x19 = fork(shift, identity, x18)
    x20 = mapply(x19, x13)
    x21 = paint(I, x20)
    return x21