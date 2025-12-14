import sys
from pathlib import Path

# Ensure parent dir (re-arc) is on sys.path so dsl/utils resolve when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dsl import *
from utils import *

def generate_bbc9ae5d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 15))
    w = w * 2
    locinv = unifint(diff_lb, diff_ub, (2, w))
    locj = w - locinv
    loc = (0, locj)
    c1 = choice(cols)
    remcols = remove(c1, cols)
    ln1 = connect((0, 0), (0, locj))
    remobj = connect((0, locj+1), (0, w - 1))
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    remobj = {(choice(ccols), ij) for ij in remobj}
    gi = canvas(-1, (1, w))
    go = canvas(-1, (w//2, w))
    ln2 = shoot(loc, (1, 1))
    gi = fill(gi, c1, ln1)
    gi = paint(gi, remobj)
    go = fill(go, c1, mapply(rbind(shoot, (0, -1)), ln2))
    for c, ij in remobj:
        go = fill(go, c, shoot(ij, (1, 1)))
    return {'input': gi, 'output': go}