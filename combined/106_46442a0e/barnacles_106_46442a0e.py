"""
# [107] 469497ad.json
* image_resizing
* draw_line_from_point
* diagonals
"""

z=lambda g:[*map(list,zip(*g[::-1]))]
def p(g):m=z(g);g=[g[r]+m[r] for r in range(len(g))];return g+z(z(g))