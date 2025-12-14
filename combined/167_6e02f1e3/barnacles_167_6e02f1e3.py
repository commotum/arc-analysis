"""
# [168] 6e19193c.json
* draw_line_from_point
* direction_guessing
* diagonals
"""

p=lambda j:[[[5,5,5],[0,0,0],[0,0,0]],[[5,0,0],[0,5,0],[0,0,5]],[[0,0,5],[0,5,0],[5,0,0]]][len(set(v for r in j for v in r))-1]