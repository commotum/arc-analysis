"""
# [201] 846bdb03.json
* pattern_moving
* pattern_reflection
* crop
* color_matching
* x_marks_the_spot
"""

def p(j):
 A,c,E,k=10,enumerate,range,0
 for W,l in c(j):
  for J,a in c(l):
   if a%5:
    for C in E(J,A,2):
     for e in E(W+1):j[e][C]=a
    for C in E(J+1,A,2):j[k*(A-1)][C]=5;k^=1
    return j