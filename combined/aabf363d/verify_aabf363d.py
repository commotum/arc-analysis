import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_aabf363d(I: Grid) -> Grid:
    x0 = fork(multiply, height, width)
    x1 = lbind(ofcolor, I)
    x2 = palette(I)
    x3 = compose(x0, x1)
    x4 = argmax(x2, x3)
    x5 = leastcolor(I)
    x6 = palette(I)
    x7 = remove(x4, x6)
    x8 = other(x7, x5)
    x9 = replace(I, x5, x4)
    x10 = replace(x9, x8, x5)
    return x10