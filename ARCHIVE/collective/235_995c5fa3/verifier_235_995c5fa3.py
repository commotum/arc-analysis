import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_995c5fa3(I: Grid) -> Grid:
    x0 = width(I)
    x1 = increment(x0)
    x2 = divide(x1, FIVE)
    x3 = astuple(FOUR, FOUR)
    x4 = canvas(NEG_ONE, x3)
    x5 = asindices(x4)
    x6 = rbind(toobject, I)
    x7 = lbind(shift, x5)
    x8 = compose(x6, x7)
    x9 = multiply(x2, FIVE)
    x10 = interval(ZERO, x9, FIVE)
    x11 = apply(tojvec, x10)
    x12 = apply(x8, x11)
    x13 = matcher(numcolors, ONE)
    x14 = fork(equality, identity, hmirror)
    x15 = compose(flip, x14)
    x16 = lbind(index, I)
    x17 = compose(x16, ulcorner)
    x18 = lbind(add, DOWN)
    x19 = chain(x16, x18, ulcorner)
    x20 = fork(equality, x17, x19)
    x21 = compose(flip, x20)
    x22 = fork(either, x13, x15)
    x23 = fork(either, x22, x21)
    x24 = compose(flip, x23)
    x25 = lbind(multiply, TWO)
    x26 = compose(x25, x13)
    x27 = lbind(multiply, FOUR)
    x28 = compose(x27, x15)
    x29 = fork(add, x26, x28)
    x30 = lbind(multiply, THREE)
    x31 = compose(x30, x21)
    x32 = lbind(multiply, EIGHT)
    x33 = compose(x32, x24)
    x34 = fork(add, x31, x33)
    x35 = fork(add, x29, x34)
    x36 = apply(x35, x12)
    x37 = rbind(repeat, x2)
    x38 = apply(x37, x36)
    return x38