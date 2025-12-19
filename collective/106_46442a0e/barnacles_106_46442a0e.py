"""
# [106] 46442a0e.json
* image_repetition
* image_reflection
"""

z=lambda g:[*map(list,zip(*g[::-1]))]
def p(g):m=z(g);g=[g[r]+m[r] for r in range(len(g))];return g+z(z(g))