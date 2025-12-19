"""
# [236] 99b1bc43.json
* take_complement
* detect_wall
* separate_images
* pattern_intersection
"""

def p(j,A=range(4)):
 for c in A:
  for E in A:
   j[c][E]+=j[c+5][E]
   if j[c][E]==3:j[c][E]=0
   elif j[c][E]>0:j[c][E]=3
 return j[:4]