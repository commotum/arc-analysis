import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ec883f72(I: Grid) -> Grid:
    x0 = fork(multiply, height, width)
    x1 = partition(I)
    x2 = argmax(x1, x0)
    x3 = remove(x2, x1)
    x4 = argmax(x3, x0)
    x5 = other(x3, x4)
    x6 = palette(I)
    x7 = lrcorner(x4)
    x8 = add(x7, UNITY)
    x9 = llcorner(x4)
    x10 = add(x9, DOWN_LEFT)
    x11 = urcorner(x4)
    x12 = add(x11, UP_RIGHT)
    x13 = ulcorner(x4)
    x14 = add(x13, NEG_UNITY)
    x15 = shoot(x8, UNITY)
    x16 = shoot(x10, DOWN_LEFT)
    x17 = shoot(x12, UP_RIGHT)
    x18 = shoot(x14, NEG_UNITY)
    x19 = combine(x15, x16)
    x20 = combine(x17, x18)
    x21 = combine(x19, x20)
    x22 = color(x5)
    x23 = fill(I, x22, x21)
    return x23