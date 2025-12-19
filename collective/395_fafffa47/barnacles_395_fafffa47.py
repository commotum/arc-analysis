"""
# [395] fafffa47.json
* separate_images
* take_complement
* pattern_intersection
"""

def p(g):t,b=g[:3],g[3:];return[[2if t[r][c]==b[r][c]==0else 0for c in range(3)]for r in range(3)]