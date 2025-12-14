import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_1caeab9d(I: Grid) -> Grid:
    x0 = objects(I, T, T, T)
    x1 = ofcolor(I, ONE)
    x2 = lowermost(x1)
    x3 = lbind(subtract, x2)
    x4 = chain(toivec, x3, lowermost)
    x5 = fork(shift, identity, x4)
    x6 = merge(x0)
    x7 = cover(I, x6)
    x8 = mapply(x5, x0)
    x9 = paint(x7, x8)
    return x9