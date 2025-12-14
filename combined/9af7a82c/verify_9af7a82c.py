import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_9af7a82c(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = order(x0, size)
    x2 = valmax(x0, size)
    x3 = rbind(astuple, ONE)
    x4 = lbind(subtract, x2)
    x5 = compose(x3, size)
    x6 = chain(x3, x4, size)
    x7 = fork(canvas, color, x5)
    x8 = lbind(canvas, ZERO)
    x9 = compose(x8, x6)
    x10 = fork(vconcat, x7, x9)
    x11 = compose(cmirror, x10)
    x12 = apply(x11, x1)
    x13 = merge(x12)
    x14 = cmirror(x13)
    return x14