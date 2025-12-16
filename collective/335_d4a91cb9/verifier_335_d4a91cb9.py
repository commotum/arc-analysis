import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_d4a91cb9(I: Grid) -> Grid:
    x0 = ofcolor(I, EIGHT)
    x1 = ofcolor(I, TWO)
    x2 = first(x0)
    x3 = first(x1)
    x4 = last(x2)
    x5 = first(x3)
    x6 = astuple(x5, x4)
    x7 = connect(x6, x2)
    x8 = connect(x6, x3)
    x9 = combine(x7, x8)
    x10 = underfill(I, FOUR, x9)
    return x10