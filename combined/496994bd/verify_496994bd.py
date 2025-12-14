import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_496994bd(I: Grid) -> Grid:
    x0 = mostcolor(I)
    x1 = vsplit(I, TWO)
    x2 = apply(numcolors, x1)
    x3 = contained(ONE, x2)
    x4 = branch(x3, hmirror, vmirror)
    x5 = x4(I)
    x6 = asobject(x5)
    x7 = matcher(first, x0)
    x8 = compose(flip, x7)
    x9 = sfilter(x6, x8)
    x10 = paint(I, x9)
    return x10