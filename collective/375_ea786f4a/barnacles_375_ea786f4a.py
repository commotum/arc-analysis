"""
# [375] ea786f4a.json
* pattern_modification
* draw_line_from_point
* diagonals
"""

def p(j):
 for A in range(len(j)):j[A][A]=j[-A-1][A]=0
 return j