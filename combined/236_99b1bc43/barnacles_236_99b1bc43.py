"""
# [237] 99fa7670.json
* draw_line_from_point
* pattern_expansion
"""

def p(j,A=range(4)):
 for c in A:
  for E in A:
   j[c][E]+=j[c+5][E]
   if j[c][E]==3:j[c][E]=0
   elif j[c][E]>0:j[c][E]=3
 return j[:4]