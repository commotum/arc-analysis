"""
# [232] 97999447.json
* draw_line_from_point
* pattern_expansion
"""

p=lambda g:[[g[i%5][j%6]for j in range(len(g[0])*2)]for i in range(len(g)*1)]