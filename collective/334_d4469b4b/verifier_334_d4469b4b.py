import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d4469b4b(I: Grid) -> Grid:
    x0 = palette(I)
    x1 = contained(ONE, x0)
    x2 = contained(TWO, x0)
    x3 = branch(x1, UNITY, TWO_BY_TWO)
    x4 = branch(x2, RIGHT, x3)
    x5 = fork(combine, vfrontier, hfrontier)
    x6 = x5(x4)
    x7 = canvas(ZERO, THREE_BY_THREE)
    x8 = fill(x7, FIVE, x6)
    return x8