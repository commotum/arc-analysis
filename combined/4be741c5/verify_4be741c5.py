import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_4be741c5(I: Grid) -> Grid:
    x0 = first(I)
    x1 = dedupe(x0)
    x2 = size(x1)
    x3 = equality(x2, ONE)
    x4 = branch(x3, dmirror, identity)
    x5 = branch(x3, height, width)
    x6 = x5(I)
    x7 = astuple(ONE, x6)
    x8 = x4(I)
    x9 = crop(x8, ORIGIN, x7)
    x10 = apply(dedupe, x9)
    x11 = x4(x10)
    return x11