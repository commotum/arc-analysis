"""
# [008] 05f2a901.json
* pattern_moving
* direction_guessing
* bring_patterns_close
"""

def p(g):R=range;L=len;d={(i+j)%3:c for i in R(L(g))for j in R(L(g[0]))for c in[g[i][j]]if c};return[[d.get((i+j)%3,0)for j in R(L(g[0]))]for i in R(L(g))]