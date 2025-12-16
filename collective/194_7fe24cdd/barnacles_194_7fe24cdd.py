"""
# [195] 80af3007.json
* crop
* pattern_resizing
* image_resizing
* fractal_repetition
"""

j=lambda A:[[*i]for i in zip(*A[::-1])]
p=lambda c:[a+b for a,b in zip(c,j(c))]+[a+b for a,b in zip(j(j(j(c))),j(j(c)))]