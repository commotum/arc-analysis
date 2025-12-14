"""
# [393] f8ff0b80.json
* separate_shapes
* count_tiles
* summarize
* order_numbers
"""

L=len
R=range
def p(g):
 C=max(sum(g,[]))
 g=[[5 if c==0 else c for c in r] for r in g]
 return g