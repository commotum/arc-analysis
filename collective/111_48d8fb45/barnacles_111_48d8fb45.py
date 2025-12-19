"""
# [111] 48d8fb45.json
* find_the_intruder
* crop
"""

p=lambda g:next([g[i+k][j-1:j+2]for k in(1,2,3)]for i,r in enumerate(g)for j,x in enumerate(r)if x==5)