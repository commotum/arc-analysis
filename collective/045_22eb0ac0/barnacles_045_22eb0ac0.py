"""
# [046] 234bbc79.json
* recoloring
* bring_patterns_close
* crop
"""

def p(j):
	for A in j:
		for c in{*A}-{0}:
			E=A.index(c);k=len(A)-A[::-1].index(c)
			for W in range(E,k):
				if~A[W]:A[W]=c
	return j