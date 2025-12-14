import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_662c240a(I: Grid) -> Grid:
    x0 = portrait(I)
    x1 = branch(x0, vsplit, hsplit)
    x2 = shape(I)
    x3 = maximum(x2)
    x4 = minimum(x2)
    x5 = divide(x3, x4)
    x6 = x1(I, x5)
    x7 = fork(equality, identity, dmirror)
    x8 = compose(flip, x7)
    x9 = extract(x6, x8)
    return x9