import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *

def verify_ce602527(I: Grid) -> Grid:
    x0 = fgpartition(I)
    x1 = rbind(bordering, I)
    x2 = extract(x0, x1)
    x3 = remove(x2, x0)
    x4 = totuple(x3)
    x5 = first(x4)
    x6 = last(x4)
    x7 = color(x5)
    x8 = mostcolor(I)
    x9 = shape(x5)
    x10 = canvas(x8, x9)
    x11 = normalize(x5)
    x12 = paint(x10, x11)
    x13 = upscale(x12, TWO)
    x14 = shape(x6)
    x15 = canvas(x8, x14)
    x16 = normalize(x6)
    x17 = paint(x15, x16)
    x18 = upscale(x17, TWO)
    x19 = shape(x2)
    x20 = canvas(x8, x19)
    x21 = normalize(x2)
    x22 = paint(x20, x21)
    x23 = color(x2)
    x24 = replace(x22, x23, x7)
    x25 = asobject(x24)
    x26 = occurrences(x13, x25)
    x27 = size(x26)
    x28 = positive(x27)
    x29 = downscale(x13, TWO)
    x30 = downscale(x18, TWO)
    x31 = branch(x28, x29, x30)
    return x31