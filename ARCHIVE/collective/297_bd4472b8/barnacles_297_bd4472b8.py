"""
# [297] bd4472b8.json
* detect_wall
* pattern_expansion
* ex_nihilo
* color_guessing
* color_palette
"""

def p(j):
 A,c=len(j),len(j[0]);E=j[0]*20
 for k in range(2,A):j[k]=[E[k-2]for _ in range(c)]
 return j