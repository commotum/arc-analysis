import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_4938f0c2(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = matcher(size, FOUR)
    x2 = fork(both, square, x1)
    x3 = extract(x0, x2)
    x4 = color(x3)
    x5 = merge(x0)
    x6 = compose(hmirror, vmirror)
    x7 = initset(x6)
    x8 = insert(vmirror, x7)
    x9 = insert(hmirror, x8)
    x10 = rapply(x9, x5)
    x11 = ulcorner(x3)
    x12 = lbind(subtract, x11)
    x13 = matcher(first, x4)
    x14 = rbind(sfilter, x13)
    x15 = chain(x12, ulcorner, x14)
    x16 = fork(shift, identity, x15)
    x17 = mapply(x16, x10)
    x18 = paint(I, x17)
    return x18