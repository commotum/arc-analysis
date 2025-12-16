"""
# [396] fcb5c309.json
* rectangle_guessing
* separate_images
* count_tiles
* take_maximum
* crop
* recoloring
"""

def p(g):t,b=g[:3],g[3:];return[[2if t[r][c]==b[r][c]==0else 0for c in range(3)]for r in range(3)]