import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_2bcee788(I: Grid) -> Grid:
    x0 = partition(I)
    x1 = fork(multiply, height, width)
    x2 = argmax(x0, x1)
    x3 = remove(x2, x0)
    x4 = argmin(x3, size)
    x5 = argmax(x3, size)
    x6 = hmatching(x4, x5)
    x7 = branch(x6, vmirror, hmirror)
    x8 = x7(x5)
    x9 = branch(x6, leftmost, uppermost)
    x10 = branch(x6, tojvec, toivec)
    x11 = x9(x4)
    x12 = x9(x5)
    x13 = greater(x11, x12)
    x14 = double(x13)
    x15 = decrement(x14)
    x16 = x10(x15)
    x17 = shape(x5)
    x18 = multiply(x16, x17)
    x19 = shift(x8, x18)
    x20 = fill(I, THREE, x2)
    x21 = paint(x20, x19)
    return x21