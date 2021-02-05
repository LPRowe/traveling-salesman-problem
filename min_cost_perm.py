import functools
import math
import string
import random

def min_cost_perm(s, cost):
    """
    Find the minimum cost permutation of s.
    The cost of placing letter i after j is cost[i][j] where 'a' maps to 0, 'b' to 1, ..., 'z' to 25
    """
    
    @functools.lru_cache(None)
    def helper(i, used):
        nonlocal s, cost
        if used == target:
            return 0
        best = math.inf
        v = set()
        for k in range(len(s)):
            if not (bitmask[k] & used) and (s[k] not in v):
                v.add(s[k])
                j = ord(s[k]) - 97
                best = min(best, cost[i][j] + helper(j, used | bitmask[k]))
        return best
    
    bitmask = [1 << i for i in range(len(s))]
    target = (1 << len(s)) - 1
    return min(helper(i, bitmask[i]) for i in range(len(s)))
    
if __name__ == "__main__":
    
    # Example: expect 8
    s = "abcde"
    cost = [[0, 5, 1, 5, 3],
            [4, 0, 9, 4, 2],
            [7, 9, 0, 10, 7],
            [1, 2, 8, 0, 2],
            [3, 9, 7, 7, 0]]
    print(min_cost_perm(s, cost))
    
    # Edge Case:
    N = 20
    letters = string.ascii_lowercase[:N]
    s = ''.join([random.choice(letters) for _ in range(N)])
    cost = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                cost[i][j] = random.randint(1, 10**9)
    print(min_cost_perm(s, cost))
    