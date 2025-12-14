import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_57aa92db(I: Grid) -> Grid:
    x0 = objects(I, F, T, T)
    x1 = palette(I)
    x2 = lbind(sfilter, x0)
    x3 = rbind(compose, palette)
    x4 = lbind(lbind, contained)
    x5 = chain(x2, x3, x4)
    x6 = compose(size, x5)
    x7 = argmax(x1, x6)
    x8 = rbind(colorcount, x7)
    x9 = apply(x8, x0)
    x10 = minimum(x9)
    x11 = rbind(colorcount, x7)
    x12 = matcher(x11, x10)
    x13 = sfilter(x0, x12)
    x14 = argmax(x13, size)
    x15 = matcher(first, x7)
    x16 = rbind(sfilter, x15)
    x17 = lbind(upscale, x14)
    x18 = chain(x17, width, x16)
    x19 = compose(ulcorner, x16)
    x20 = chain(ulcorner, x16, x18)
    x21 = fork(subtract, x19, x20)
    x22 = fork(shift, x18, x21)
    x23 = rbind(other, x7)
    x24 = compose(x23, palette)
    x25 = compose(x16, x22)
    x26 = fork(difference, x22, x25)
    x27 = fork(recolor, x24, x26)
    x28 = compose(x16, x22)
    x29 = fork(combine, x28, x27)
    x30 = remove(x14, x0)
    x31 = mapply(x29, x30)
    x32 = paint(I, x31)
    return x32