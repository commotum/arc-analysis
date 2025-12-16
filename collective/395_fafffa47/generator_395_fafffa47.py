import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl/utils resolve when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *
from utils import *

def generate_fafffa47(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    res = set(inds) - (set(aset) | set(bset))
    go = fill(c, 2, res)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}