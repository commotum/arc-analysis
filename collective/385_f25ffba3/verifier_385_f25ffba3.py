import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_f25ffba3(I: Grid) -> Grid:
    x0 = tophalf(I)
    x1 = numcolors(x0)
    x2 = equality(x1, ONE)
    x3 = bottomhalf(I)
    x4 = numcolors(x3)
    x5 = equality(x4, ONE)
    x6 = either(x2, x5)
    x7 = branch(x6, identity, dmirror)
    x8 = x7(I)
    x9 = asobject(x8)
    x10 = hmirror(x9)
    x11 = mostcolor(I)
    x12 = matcher(first, x11)
    x13 = compose(flip, x12)
    x14 = sfilter(x10, x13)
    x15 = paint(x8, x14)
    x16 = x7(x15)
    return x16