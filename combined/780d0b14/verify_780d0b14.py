import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_780d0b14(I: Grid) -> Grid:
    x0 = frontiers(I)
    x1 = merge(x0)
    x2 = color(x1)
    x3 = merge(x0)
    x4 = fill(I, NEG_ONE, x3)
    x5 = shape(I)
    x6 = canvas(NEG_ONE, x5)
    x7 = hconcat(x4, x6)
    x8 = objects(x7, F, F, T)
    x9 = rbind(other, x2)
    x10 = compose(x9, palette)
    x11 = fork(astuple, x10, ulcorner)
    x12 = apply(x11, x8)
    x13 = merge(x8)
    x14 = fill(I, x2, x13)
    x15 = paint(x14, x12)
    x16 = compress(x15)
    return x16