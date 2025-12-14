import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_72322fa7(I: Grid) -> Grid:
    x0 = objects(I, F, T, T)
    x1 = matcher(numcolors, TWO)
    x2 = sfilter(x0, x1)
    x3 = apply(normalize, x2)
    x4 = chain(first, totuple, palette)
    x5 = chain(last, totuple, palette)
    x6 = lbind(matcher, first)
    x7 = compose(x6, x4)
    x8 = lbind(matcher, first)
    x9 = compose(x8, x5)
    x10 = fork(sfilter, identity, x7)
    x11 = fork(sfilter, identity, x9)
    x12 = lbind(occurrences, I)
    x13 = chain(invert, ulcorner, x10)
    x14 = chain(invert, ulcorner, x11)
    x15 = lbind(lbind, shift)
    x16 = fork(shift, identity, x13)
    x17 = fork(shift, identity, x14)
    x18 = compose(x15, x16)
    x19 = compose(x12, x10)
    x20 = fork(mapply, x18, x19)
    x21 = compose(x15, x17)
    x22 = compose(x12, x11)
    x23 = fork(mapply, x21, x22)
    x24 = fork(combine, x20, x23)
    x25 = mapply(x24, x3)
    x26 = paint(I, x25)
    return x26