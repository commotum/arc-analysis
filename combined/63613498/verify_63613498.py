import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_63613498(I: Grid) -> Grid:
    x0 = objects(I, T, F, T)
    x1 = mostcolor(I)
    x2 = fork(add, height, width)
    x3 = compose(decrement, x2)
    x4 = fork(equality, x3, size)
    x5 = rbind(bordering, I)
    x6 = fork(both, x4, x5)
    x7 = rbind(toobject, I)
    x8 = chain(numcolors, x7, delta)
    x9 = matcher(x8, TWO)
    x10 = fork(both, x6, x9)
    x11 = sfilter(x0, x10)
    x12 = argmax(x11, size)
    x13 = delta(x12)
    x14 = toobject(x13, I)
    x15 = matcher(first, x1)
    x16 = compose(flip, x15)
    x17 = sfilter(x14, x16)
    x18 = normalize(x17)
    x19 = toindices(x18)
    x20 = compose(toindices, normalize)
    x21 = matcher(x20, x19)
    x22 = remove(x17, x0)
    x23 = argmax(x22, x21)
    x24 = color(x12)
    x25 = fill(I, x24, x23)
    return x25