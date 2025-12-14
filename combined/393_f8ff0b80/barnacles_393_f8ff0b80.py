"""
# [394-R] f9012d9b.json
* pattern_expansion
* pattern_completion
* crop
"""

p=lambda j:[[k]for k,_ in __import__('collections').Counter(i for r in j for i in r).most_common(4)[1:]]