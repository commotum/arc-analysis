"""
# [348] db3e9e38.json
* pattern_expansion
* out_of_boundary
"""

def p(j,A=range(3)):
 for c in A:
  for E in A:
   j[c][E]+=j[c][E+3]
   if j[c][E]>0:j[c][E]=6
 return[R[:3]for R in j]