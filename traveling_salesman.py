"""LICENSE
This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree. 
"""

"""INTRO
The traveling salesman problem (TSP) requires a salesman to visit
all n vertices (nodes) and return to the starting node taking the
shortest possible path.  

The brute force solution is O(n!) and becomes intractible above
approximately 12 nodes and the dynamic programming solution
is O(n**2 2**n) and is feasible for up to approximately 20 nodes.

However many instances of the TSP have hundreds or thousands of nodes.
In such cases we must settle for an approximate solution.

Here we will solve the TSP optimally and using a 3 heuristics, each one
improving on it's predecessor.  

Summary:
    1. heuristic_path    O(n**2) MST and preorder traversal of points
    2. relaxed_heur_path O(n**2) relax points from heuristic_path solution
    3. k_optimized_path  O(n k**2 2**k) where k = n // 2 
       optimize continuous subpaths of k nodes from relaxed_heur_path solution

Approximate Error:
    1. 14% over optimal path
    2. 5% over optimal path
    3. 0% over optimal path
"""

import collections
import functools
import heapq
import math
import random
import time

import matplotlib
import matplotlib.pyplot as plt

# =============================================================================
# CUSTOM CLASSES
# =============================================================================

class UnionFind:
    def __init__(self):
        self.group_id = 0
        self.groups = {}
        self.id = {}
        
    def union(self, a, b):
        """Returns True if an edge was created"""
        A, B = a in self.id, b in self.id
        if A and B and self.id[a] != self.id[b]:
            self.merge(a, b)
        elif A and B:
            return False
        elif A or B:
            self.add(a, b)
        else:
            self.create(a, b)
        return True
        
    def merge(self, a, b):
        """City a and city b belong to different groups, merge the smaller group with the larger group"""
        obs, targ = sorted((self.id[a], self.id[b]), key = lambda i: len(self.groups[i]))
        for node in self.groups[obs]:
            self.id[node] = targ
        self.groups[targ] |= self.groups[obs]
        del self.groups[obs]
        
    def add(self, a, b):
        """City a or city b does not belong to a group, add the new city to the existing group."""
        a, b = (a, b) if a in self.id else (b, a)
        targ = self.id[a]
        self.id[b] = targ
        self.groups[targ] |= {b}
        
    def create(self, a, b):
        """Neither city a nor city b belongs to a group, create a new group {a, b}"""
        self.groups[self.group_id] = {a, b}
        self.id[a] = self.id[b] = self.group_id
        self.group_id += 1
        
# =============================================================================
# CUSTOM WRAPPERS        
# =============================================================================

def timer(fcn):
    """Returns the runtime in ms of the function fcn"""
    def wrapper(*args, **kwargs):
        t0 = time.time_ns()
        res = fcn(*args, **kwargs)
        t = time.time_ns() - t0
        print(f"{fcn.__name__}: {round(t * 10**-6, 1)} 'ms'")
        return res
    return wrapper

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
    
def distance(start, fin):
    """
    Returns the euclidean distnace between start (x1, y1) and fin (x2, y2).
    start: (float, float)
    fin: (float, float)
    returns float
    """
    x1, y1 = start
    x2, y2 = fin
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def generate_points(n, bound):
    """
    Generates a random list of n city positions (xi, yi).
    n: number of cities
    bound: maximum allowed X or Y position of any city
    returns List[(x1, y1), ..., (xn, yn)]
    """
    points = set()
    while len(points) < n:
        points.add((bound * random.random(), bound * random.random()))
    return list(points)
    
def path_cost(points):
    """
    Returns the length of the path that traverses all points.
    points List[(x1, y1), ..., (xn, yn), (x1, y1)] 
    """
    return sum(distance(a, b) for a, b in zip(points, points[1:]))

# =============================================================================
# TRAVELING SALESMAN OPTIMAL SOLUTION ALGORITHMS
# =============================================================================

@timer
def optimal_path_dp(points):
    """
    Finds the optimal path to visit all of points and return to the starting point.
    Time Complexity: O(n**2 2**n)
    points: List[(x1, y1), ..., (xn, yn)]
    returns (length_of_path, path)
    length_of_path: float
    path: List[(x1, y1), ..., (xn, yn), (x1, y1)]
    """
    
    @functools.lru_cache(None)
    def helper(i, used):
        
        if used == target:
            return (cost[i][0], [points[0]])
        
        best = math.inf
        best_path = []
        for j in range(N):
            if not (used & bitmask[j]):
                c, p = helper(j, used | bitmask[j])
                c += cost[i][j]
                if c < best:
                    best = c
                    best_path = [points[j]] + p
        return best, best_path
    
    N = len(points)
    cost = [[distance(points[i], points[j]) for i in range(N)] for j in range(N)]
    bitmask = [1 << i for i in range(N)]
    target = (1 << N) - 1
    c, p = helper(0, 1)
    return c, [points[0]] + p
    
@timer
def optimal_path(points):
    """
    Time complexity: O(n!) For edmonstration purposes only!
    
    Non-memoized approach to finding the optimal path.
    Takes a list of points and finds the optimal tour to visit all points and return home.
    Very slow compared to memoized version.
    
    points: List[(x1, y1), ..., (xn, yn)] where xn, yn are floats
    returns cost_of_best_path, best_path where cost_of_best_path is a float and best_path
            is [(x1, y1), ..., (xn, yn), (x1, y1)] 
    """
    
    def helper(cost, path, points):
        nonlocal best_path, best
        
        if not points or cost >= best:
            cost += distance(path[-1], path[0])
            if cost < best:
                path.append(path[0])
                best = cost
                best_path = path
            return None
        
        for i,p in enumerate(points):
            helper(cost + distance(path[-1], p), path + [p], points[:i]+points[i+1:])
    
    best = math.inf
    best_path = []
    for i, p in enumerate(points):
        helper(0, [p], points[:i]+points[i+1:])
    return best, best_path

# =============================================================================
# TRAVELING SALESMAN HEURISTIC SOLUTION ALGORITHMS
# =============================================================================

@timer
def heuristic_path(points):
    """
    Uses Kruskal's algorithm to create a MST of the points
    Performs a pre-order DFS exploration from each node and creates a path as the nodes are visited.
    Selects the shortest path to visit all nodes and return home.
    
    points: List[(x1, y1), ..., (xn, yn)]
    returns edge_list_of_MST, cost_of_best_path, best_path
    
    edge_list_of_MST: [(p1, p2), (p1, p3), (p3, p4), ...] where pi is position (xi, yi)
    cost_of_best_path: float, length of the best path
    best_path: List[(x1, y1), ..., (xn, yn), (x1, y1)]
    """
    
    # Store edges in a min heap
    h = []
    for i in range(len(points)):
        a = points[i]
        for j in range(i):
            b = points[j]
            heapq.heappush(h, (distance(a, b), a, b))
    
    # Create MST
    uf = UnionFind()
    edges = []
    while len(edges) < len(points) - 1:
        dist, a, b = heapq.heappop(h)
        if uf.union(a, b):
            edges.append((a, b))
    
    # Convert edge list to undirected graph
    g = collections.defaultdict(list)
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)
    
    # Perform preorder traversal to find approximation of the optimal path
    def helper(node):
        nonlocal path, best_path, best, cost, visited
        path.append(node)
        for neigh in g[node]:
            if neigh not in visited:
                visited.add(neigh)
                cost += distance(neigh, node)
                helper(neigh)
    
    # Try using each node as a starting node for the preorder traversal
    best_path = []
    best = math.inf
    for start in points:
        visited = set([start])
        path = []
        cost = 0
        helper(start)
        path.append(start)
        cost = sum(distance(a, b) for a, b in zip(path, path[1:]))
        if cost < best:
            best = cost
            best_path = path
    
    return edges, best, best_path

def path_relaxation(points):
    """
    points: the tour suggested by heuristic_path List[(x1, y1), ..., (xn, yn), (x1, y1)]
    
    note: points[0] = points[-1] = the starting city, never move points[0] or points[-1]
    
    Tries removing one node from the path and tries to insert it between all other
    cities.  The location that results in the smallest total path is chosen.
    This process is repeated for all n nodes. (consider prioritizing the order of checking nodes)
    i.e. do most streched node first
    
    returns cost_of_path, List[(x1, y1), ..., (xn, yn), (x1, y1)]
    """
    
    def reduced_cost(i):
        """returns the change in path length when point i is removed from the path
        reduced_cost should always be <= 0 when triangulation inequality is true (it is for geometric points)"""
        a, b, c = points[i-1], points[i], points[i+1]
        return distance(a, c) - distance(a, b) - distance(b, c)
    
    def insertion_cost(a, b, c):
        """returns the cost of inserting point b in between points a and c
        insertion_cost should always be >= 0 when triangulation inequality is true (it is for geometric points)"""
        return distance(a, b) + distance(b, c) - distance(a, c)
    
    adjusted = set() # keep track of which points have already been adjusted bc order of points may change
    while len(adjusted) < len(points) - 2:
        for i in range(1, len(points) - 1):
            if points[i] not in adjusted:
                adjusted.add(points[i])
                rc = reduced_cost(i)   # change in cost by removing point i
                best = 0
                best_index = i
                for j in range(len(points) - 1):
                    if j != i and j != i - 1:
                        c = insertion_cost(points[j], points[i], points[j+1]) # increase in cost by adding point i
                        total_cost = c + rc
                        if total_cost < best:
                            best = total_cost
                            best_index = j
                
                # update the order in which points[i] is visited if a lower cost order was found
                if best < 0 and best_index != i:
                    j = best_index
                    p = [points[i]]
                    if j < i:
                        points = points[:j+1] + p + points[j+1:i] + points[i+1:]
                    else:
                        points = points[:i] + points[i+1:j+1] + p + points[j+1:]

    return path_cost(points), points

@timer
def relaxed_heur_path(points):
    """
    Calculates the heuristic path using Kruskal's Algorithm for MST and pre-order traversals
    Then relaxes the path by removing and reinserting points at more optimal locations in the path
    to remove tension from the path
    
    Returns edges of the MST, cost of the relaxed path, and the relaxed path
    """
    edges, heur_cost, heur_best_path = heuristic_path(points)
    rel_cost, relaxed_path = path_relaxation(heur_best_path)
    prev = heur_cost
    while rel_cost != prev:
        prev = rel_cost
        rel_cost, relaxed_path = path_relaxation(relaxed_path)
    return edges, rel_cost, relaxed_path

@timer
def k_optimized_path(points, k):
    """
    Uses the relaxed heuristic path method to approximate the best TSP path.
    Then optimizes subsets (of size k) of the path
    points: List[(x1, y1), ..., (xn, yn), (x1, y1)]
    k: int, 1 <= k <= len(points) - 1
    
    recommended n // 4 <= k <= n // 2 where n is number of points
    smaller k is less likely to find optimal path but greatly decreases time complexity
    k = n // 2 is likely to find the optimal path, but approaches the dynamic programming time complexity
    
    Time complexity: O(n k**2 2**k)
    Note: k > 20 is intractible, for reasonable walltime keep k <= 10
    """
    edges, rel_cost, relaxed_path = relaxed_heur_path(points)
    cost, points = partial_tour_optimization(relaxed_path, k)
    return cost, points

@timer
def partial_tour_optimization(points, k):
    """
    points: list of (x, y) coordinates of the path that visits all nodes
    k: length of the subset of points that should be optimized
    
    Considers points[i:j+1] where point[i] and point[j] are pinned at their location in the tour.
    The path from points[i] to points[j] is then replaced with an optimal path
    from points[i] -> points[i+1:j] -> points[j].
    
    returns tour_cost, optimized_tour
    """
    
    @functools.lru_cache(None)
    def helper(p, used):
        """p is index of previous node, used is bitmask of used indices"""
        nonlocal points, target, i, j
        
        if used == target:
            return (distance(points[p], points[j]), [points[j]])
        
        best = math.inf
        best_path = []
        for m in range(i+1, j):
            if not (used & bitmask[m]):
                c, path = helper(m, used | bitmask[m])
                #print(points[p])
                c += distance(points[p], points[m])
                if c < best:
                    best = c
                    best_path = [points[m]] + path
        return best, best_path


    points.pop() # remove second start point
    N = len(points)
    points = 2 * points # double to handle circular aspect more easily
    bitmask = [1 << i for i in range(len(points))]
    for i in range(len(points) // 2):
        j = i + k + 1
        target = sum((1 << m for m in range(i+1, j)))
        cost, path_ = helper(i, 0)
        helper.cache_clear()
        n = 0
        for m in range(i+1, j):
            points[m] = path_[n]
            if m + N < len(points):
                points[m + N] = path_[n]
            n += 1

    points = points[N:] + [points[N]] # add start point
    
    return path_cost(points), points

# =============================================================================
# ACCURACY ASSESSMENT FUNCTIONS
# =============================================================================

def average_error(func1, func2, n = 12, cycles = 10):
    """
    Compares the path length calculated by func2 and func1. This assumes that func1 produces
    the optimal path length and func2 is heuristic_path or relaxed_heur_path.
    
    n: number of nodes
    cycles: how many times to generate a new set of nodes and run func1 and func2
    
    returns None
    prints (sum of optimal path lengths, sum of approx path lengths, overestimate percentage)
    """
    best = heur = 0
    for i in range(cycles):
        print(i, '/', cycles)
        points = generate_points(n, 100)
        best += func1(points)[0]
        heur += func2(points)[1]
    print(round(best, 1), round(heur, 1), round(100*(heur - best) / best, 2))
    
def average_error2(func1, func2, k, n = 12, cycles = 10):
    """
    Compares the path length calculated by func2 and func1. This assumes that func1 produces
    the optimal path length and func2 is k_optimized_path.
    
    n: number of nodes
    cycles: how many times to generate a new set of nodes and run func1 and func2
    
    returns None
    prints (sum of optimal path lengths, sum of approx path lengths, overestimate percentage)
    """
    best = heur = 0
    for i in range(cycles):
        print(i, '/', cycles)
        points = generate_points(n, 100)
        best += func1(points)[0]
        heur += func2(points, k)[0]
    print(n, int(best), int(heur), round(100*(heur - best) / best, 3))

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_path(points, color, style = '-', connect_the_dots = True, fig_num = 1, line_width = 1, dot_size = 4):
    """Plots a list of (x, y) coordinates on a scatter plot and connects the path with a line."""
    plt.figure(fig_num, figsize = (3.2, 2.4), dpi = 300)
    x, y = list(zip(*points))
    plt.scatter(x, y, color = color, s = dot_size)
    if connect_the_dots:
        plt.plot(x, y, color+style, lw = line_width)
    plt.show()
    
def plot_edges(edges, color, style = '--', fig_num = 2, line_width = 1):
    """
    plots the minimum spanning tree (MST) for the set of points
    edges: edge_list_of_MST, [(p1, p2), (p1, p3), (p3, p4), ...] where pi is position (xi, yi)
    """
    plt.figure(fig_num, figsize = (3.2, 2.4), dpi = 300)
    for a, b in edges:
        plt.scatter([a[0], b[0]], [a[1], b[1]], color = color, s = 4)
        plt.plot([a[0], b[0]], [a[1], b[1]], color + style, lw = line_width)
    plt.show()

def compare(points):
    """
    Takes a list of geometric points [(x1, y1), (x2, y2), ...] and finds the optimal and approxiamte
    path to visit all points.
    
    Three approximations are calculated and compared to the optimal path:
        1. heuristic_path
        2. relaxed_heur_path
        3. k_optimized_path
    """
    cost, best_path = optimal_path_dp(points)
    edges, heur_cost, heur_best_path = heuristic_path(points)
    rel_cost, relaxed_path = path_relaxation(heur_best_path)
    prev = heur_cost
    while rel_cost != prev:
        prev = rel_cost
        rel_cost, relaxed_path = path_relaxation(relaxed_path)
        
    # show best path vs heuristic path
    name = "MST and preorder"
    plot_path(best_path, 'g', fig_num = name)
    plot_path(heur_best_path, 'r', style = '--', fig_num = name)
    error = 100 * (heur_cost - cost) / cost
    plt.title(f"Optimal: {round(cost, 1)} Approx.: {round(heur_cost, 1)} Error: {round(error, 1)}%")
    plt.legend(["Optimal Path", "Approximate Path"])
    
    # show mst
    name = "MST"
    plot_edges(edges, 'b', style = '-.', fig_num = name)
    plt.title("Minimum Spanning Tree")
    
    # show best path vs heuristic path with relaxation
    name = "Path Relaxation"
    plot_path(best_path, 'g', fig_num = name)
    plot_path(relaxed_path, 'r', style = '--', fig_num = name)
    error = 100 * (rel_cost - cost) / cost
    plt.title(f"Optimal: {round(cost, 1)} Approx.: {round(rel_cost, 1)} Error: {round(error, 1)}%")
    plt.legend(["Optimal Path", "Approximate Path"])
    
    # Perform relaxation and subpath optimization
    name = "k-Optimization"
    k_opt_cost, sub_path = partial_tour_optimization(relaxed_path, n // 2)
    plot_path(best_path, 'g', fig_num = name)
    plot_path(sub_path, 'r', style = '--', fig_num = name)
    error = 100 * (k_opt_cost - cost) / cost
    plt.title(f"Optimal: {round(cost, 1)} Approx.: {round(k_opt_cost, 1)} Error: {round(error, 1)}%")
    plt.legend(["Optimal Path", "Approximate Path"])
    
    print(f"heuristic cost: {int(heur_cost)}",
          f"relaxed cost: {int(rel_cost)}",
          f"k-optimized: cost {int(k_opt_cost)}", 
          f"optimal cost: {int(cost)}", sep='\n')
    
if __name__ == "__main__":
    plt.close('all')
    matplotlib.rc('font', size = 7)
    n = 17          # number of cities in the Traveling Salesman Problem
    bound = 100     # cities locations randomly generated in (x, y) = ([0, bound], [0, bound])
    points = generate_points(n, bound)
    compare(points) # compare the 3 approxiamte solutions to the optimal solution