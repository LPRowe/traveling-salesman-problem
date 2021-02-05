### Usage

<b>traveling_salesman.py</b>
1. <b>optimal_path</b><br>
O(n!) Solves TSP optimally for a given list of (x, y) coordinates
2. <b>optimal_path_dp</b><br>
O(n<sup>2</sup> 2<sup>n</sup>) Solves TSP optimally
3. <b>heuristic_path</b><br>
O(n<sup>2</sup>) Approximate solution typ. 14% longer than optimal path for 20 cities
4. <b>relaxed_heur_path</b><br>
O(n<sup>2</sup>) Approximate solution typ. 5% longer than optimal path for 20 cities
5. <b>k_optimized_path</b><br>
O(n k<sup>2</sup> 2<sup>k</sup>) Approximate solution typ. 0% longer than optimal path for 20 cities

Note: The above functions can be applied to any list of N-dimensional coordinates that satisfy the triangle inequality theorem by updating the distance function. 

<p align="center">
<img src="./images/heur-rel-sub/20h5.png" width="32%">
<img src="./images/heur-rel-sub/20r5.png" width="32%">
<img src="./images/heur-rel-sub/20s5.png" width="32%">
<figcaption>Green Path: O(n<sup>2</sup> 2<sup>n</sup>) Optimum solution to the traveling salesman problem (TSP)<br>
Left: O(n<sup>2</sup>) Approximate solution via minimum spanning tree (MST) and preorder traversal<br>
Middle: O(n<sup>2</sup>) Improved version of left through path relaxation<br> 
Right: O(n k<sup>2</sup> 2<sup>k</sup>) Improved version of middle through k-optimization</figcaption>
</p>


### Intro

The traveling salesman problem (TSP) is a classic NP-complete problem.

The problem consists of a salesman who must travel from home, to a set of cities, and return home.  His objective is to find the shortest possible path to do so.  The problem of course does not need to be cities and roads.

Think of the problem in terms of vertices (cities) that are fully connected by a set of edges (roads).  The solution to this probelm can be applied to a wide variety of situations.

### Methods

#### <b>The brute force approach:</b>
1. generate all permuations of N cities
2. calculate the cost of the path created by each permuation
Here, the cost to travel from city A to city B is the distance between cities A and B.
However, different instances of the problem may have different cost functions.
3. return the lowest cost path
which works... but scales with O(n!) which makes this approach infeasible for sets of cities larger than &approx; 12.

The time and space complexity of the solution can be greatly improved through dynamic programming.

#### <b>The dynamic programming approach:</b>
1. Choose a specific starting city.<br>
We must travel to every city once, so it does not matter which city you choose.
2. Store the visited cities in a bit-masked integer.<br>
i.e. 106 = '0b1101010' means we have visited cities 1, 3, 5, and 6 (counting the '1' bits from the right)
3. At each iteration, consider visiting all N cities next.<br>
If a city has already been visited (as shown in the bit-mask) then do not visit it a second time.<br>
Otherwise try visiting the city for the cost of distance(current_city, next_city) plus the cost to visit the remaining cities.

This is the bones of the DP solution, but thus far it is not much better than the brute force solution.

Why? Because it is currently lacking memoization.

Consider the following scenarios for a case of 7 cities, <br>
A: the salesman visits city 1, 3, 2, 4 <br>
B: the salesman visits city 1, 2, 3, 4 

Journies A and B may have different costs, but the <b>best</b> possible path from city 4 to cities {5, 6, 7} and returning home to city 1 must be the same for both path's A and B.  Once the optimal path from 4 -> {5, 6, 7} -> 1 has been calculated once in scenario A, it would be a duplication of effort to calculate it again for scenario B.

Memoizing the minimum cost and best path from the state (current_city, visited_cities) allows us to skip these calculations the second, third, ...Nth time the state is visited.  

This reduces the time complexity of the solution from <b>O(n!)</b> to <b>O(n<sup>2</sup>&middot;2<sup>n</sup>).</b><br>
This is a substantial improvement, which allows us to calculate the optimal solution for up to &approx; 20 cities.

At this point, you might be thinking 20 cities is not a lot of cities, and you are right.
However, in order to further improve the time complexity of our solution, we must forego optimality.

#### <b>Enter the heuristic approach:</b>

1. Calculate the minimum spanning tree (MST) of the set of cities.
This can be done with Prim's Algorithm or Kruskal's Algorithm.
Kruskal's Algorithm was implemented here - see traveling_salesman.py >> heuristic_path
2. Create a path based on the order the cities are visited in a preorder traversal of the MST.
3. Repeat step 2 for all N cities and return the path that has the minimum cost.

The third step is what dominates the time complexity analysis.  Each pre-order terversal is an O(n) operation, and we are preforming the operation once for each city resulting in O(n<sup>2</sup>) time complexity.

### Comparisons

Let's take a look at a very rough estimate for the number of operations required for a set of 20 cities for each of the 3 methods mentioned so far:

| method | time complexity | approximate operations for 20 cities |
|:---:|:---:|:---:|
|Brute Force| O(n!)| &approx; 2&middot;10<sup>18</sup>|
|Dynamic Programming |O(n<sup>2</sup>2<sup>n</sup>)| &approx; 4&middot;10<sup>8</sup> | 
|Heuristic Approach | O(n<sup>2</sup>) | &approx; 800* |
<i>*Generating a list of edges for Kruskal's Algorithm is also O(n<sup>2</sup>) - could be improved.</i>

Clearly the heuristic approach is much faster than the optimal solution using dynamic programming.
However, the increase in speed comes at the cost of accuracy.  The relative accuracy of the heuristic algorithm compared to the optimal solution is given in the table below.


| n | (heuristic - optimal) / optimal | | n | (heuristic - optimal) / optimal |
|:----:|:---:|:---:|:---:|:---:|
| 3  | 0.0% | | 12  | 6.8% |
| 4  | 0.3% | | 13  | 7.3% |
| 5  | 1.0%  | | 14  | 8.5% |
| 6  | 1.6%  | | 15  | 9.0% |
| 7  | 2.3%  | | 16  | 9.6% |
| 8  | 2.8%  | | 17  | 10.5% |
| 9  | 4.3%  | | 18  | 13.1% |
| 10  | 5.4%  | | 19 | 13.6% |
| 11  | 6.2%  | | 20 | 14.0% |

<i>*{18, 19, 20} calculated from 20 samples, {3 - 17} calculated from 100 - 1000 samples.</i>

### Thoughts:

This method is looking pretty good.  In a situation where you have 20 or less nodes, sufficient memory, and it is important to have the optimal result, the dynamic programming solution is probably still the way to go.

However in situations, such as an AI player deciding on what path to follow in a game, where it is important to make quick decisions, a slightly less optimal result at the cost of 1 millisecond wall time will be preferable to an optimal result at the cost of 30 seconds wall time. 

And of course, if you have hundreds or thousands of nodes, then the heuristic method is really the only option.

But can we do better and while maintaining a time complexity of O(n<sup>2</sup>)? <b>Absolutetly.</b>

Let's compare a few of the optimal and approximate tour pairs to get an idea of where we can improve.

### Optimal versus Approximate Tours:

<p align="center"><b>10 Nodes</b><br>
<img src="./images/10_nodes_.png" width="40%">
<img src="./images/10_nodes.png" width="40%">
</p>

<p align="center"><b>14 Nodes</b><br>
<img src="./images/14_nodes_.png" width="40%">
<img src="./images/14_nodes.png" width="40%">
</p>

<p align="center"><b>18 Nodes</b><br>
<img src="./images/1.png" width="40%">
<img src="./images/1e.png" width="40%">

<img src="./images/4.png" width="40%">
<img src="./images/4e.png" width="40%">
</p>

### Observations:

1. When there are fewer nodes, the heuristic approach will often find the optimal path
2. Often, the error originates due to the path taking a detour through a point that would obviously fit better elsewhere in the path

### Idea:

Think of the path like a rubber band.  If we take out a point, the path will fill the gap by connecting the 2 neighboring points 2.

In doing so, the rubber band will relax a little because it is not being stretched as far.

However, we need to visit all the points so we must re-insert the point between two of the existing nodes in the path.

But, perhaps there is a better place in the path for this point than where we removed it from.  

Look at the approximate path for the 14 node plot.  Now picture removing the node at (42, 22) from the plot. The rubber band would relax by connecting nodes (60, 5) and (50, 35).  Insert the point (42, 22) back into the path, which two nodes should it go between?  The two nodes that will stretch the rubber band the least.  In this case (25, 21) and (50, 35).  

### Improvement to the heuristic method:

I call this improvement relaxation because it is relaxing the tension in the hypothetical rubber band.  

It is a simple and intuitive idea for improving the heuristic and as such probably already has a more formal name.

To relax the path:
1. For each node (b) calculate the cost of removing it from the path.<br>
<b>removal_cost = distance(a, c) - distance(b, c) - distance(a, b)</b> <br>
where a and c are the neighbors of node b in the path<br>
removal_cost will be negative because we are removing tension from the path by removing node b<br>
2. For each pair of neighboring nodes, calculate the cost of inserting node b<br>
<b>insertion_cost = distance(a, b) + distance(b, c) - distance(a, c)</b>
insertion cost will be non-negative because we are stretching the band to insert point b<br>
3. Insert node b at the location that will have the greatest overall relaxation for the band<br>
<b>total_relaxation = insertion_cost + removal_cost</b><br>
4. Repeat for all nodes in the path
5. Repeat steps 1 - 4 until there are no gains from path relaxation (typically 1 or 2 cycles in total)

### Comparisons

| n | (heuristic - optimal) / optimal | | n | (heuristic - optimal) / optimal |
|:----:|:---:|:---:|:---:|:---:|
| 3  | 0.00% | | 12  | 1.25% |
| 4  | 0.00% | | 13  | 1.36% |
| 5  | 0.00%  | | 14  | 2.18% |
| 6  | 0.07%  | | 15  | 2.80% |
| 7  | 0.11%  | | 16  | 3.12% |
| 8  | 0.31%  | | 17  | 3.18% |
| 9  | 0.40%  | | 18  | 4.22% |
| 10  | 0.71%  | | 19 | 4.52% |
| 11  | 1.19%  | | 20 | 4.48% |

<i>*{20} calculated from 50 samples, {3 - 19} calculated from 100 - 1000 samples.</i>

### Optimal versus Approximate Tours:
<b>Before Relaxatoin (left) and After Relaxation (right)</b>

<p align="center"><b>18 Nodes</b><br>
<img src="./images/18_pt_heur_path.png" width="40%">
<img src="./images/18_pt_relaxed.png" width="40%">
</p>


<p align="center"><b>20 Nodes</b><br>
<img src="./images/20_pt_heur_path.png" width="40%">
<img src="./images/20_pt_relaxed.png" width="40%">
</p>

<p align="center"><b>20 Nodes</b><br>
<img src="./images/20_pt_heur_path2.png" width="40%">
<img src="./images/20_pt_relaxed2.png" width="40%">
</p>

This method is looking great compared to the previous heuristic!<br>
But is there more that we can do?

### Observations:

We can learn more by studying where path relaxation failed to produce the optimal solution than we can from studying where it worked perfectly.

Firstly, let's consider what we could do to improve the solution in the 20 node scenario.  Specifically let us consider the first set of 20 nodes.

The optimal solution and relaxed heuristic solution agree on the best path everywhere except for the 8 nodes between (35, 99) and (95, 83).

Since this is a small subset of the total set of nodes, we could apply the dynamic programming solution to find the best path from node (35, 99) to node (5, 83) keeping the two nodes pinned with respect to the remainder of the nodes.  This would require O(k<sup>2</sup>&middot;2<sup>k</sup>) time where k is the length of the gap between nodes.

Furthermore because in general we will not know where the heuristic solution deviates from the optimal solution, this needs to be applied n times along the path of nodes - resulting in O(n&middot;k<sup>2</sup>&middot;2<sup>k</sup>) time complexity.  Which for 2&middot;k <= n will still be much faster than the dynamic programming approach for the optimal solution.  

### Visual result of k-subset optimization:

<b>Before relaxation (left), After relaxation (middle), After subset optimization (right)</b><br>
<b>k is the subset size that was optimized.</b>

<p align="center"><b>20 Nodes ; k = 5</b><br>
<img src="./images/heur-rel-sub/204hn5.png" width="32%">
<img src="./images/heur-rel-sub/204rn5.png" width="32%">
<img src="./images/heur-rel-sub/204sn5.png" width="32%">
</p>


<p align="center"><b>20 Nodes ; k = 8</b><br>
<img src="./images/heur-rel-sub/20h1.png" width="32%">
<img src="./images/heur-rel-sub/20r1.png" width="32%">
<img src="./images/heur-rel-sub/20s1.png" width="32%">
</p>

<p align="center"><b>20 Nodes ; k = 10</b><br>
<img src="./images/heur-rel-sub/20h2.png" width="32%">
<img src="./images/heur-rel-sub/20r2.png" width="32%">
<img src="./images/heur-rel-sub/20s2.png" width="32%">
</p>

<p align="center"><b>20 Nodes ; k = 10</b><br>
<img src="./images/heur-rel-sub/203h.png" width="32%">
<img src="./images/heur-rel-sub/203r.png" width="32%">
<img src="./images/heur-rel-sub/203s.png" width="32%">
</p>

### Comparisons

<b>Approximate path is found through:</b>
1. Calculate MST
2. Find best (lowest cost) pre-order traversal of MST
3. Relax points one by one
4. Replace continuous subsets of k nodes with the optimal arrangement where k = n // 2.

| n | (heuristic - optimal) / optimal | | n | (heuristic - optimal) / optimal |
|:----:|:---:|:---:|:---:|:---:|
| 3  | 0.000% | | 12  | 0.200% |
| 4  | 0.000% | | 13  | 0.352% |
| 5  | 0.012%  | | 14  | 0.232% |
| 6  | 0.002%  | | 15  | 0.423% |
| 7  | 0.029%  | | 16  | 0.620% |
| 8  | 0.017%  | | 17  | 0.454% |
| 9  | 0.070%  | | 18  | 0.225% |
| 10  | 0.083%  | | 19 | 0.856% |
| 11  | 0.200%  | | 20 | 0.458% |

<b><i>*{19, 20} calcualted from 50 samples, {3-18} calculated from 100 - 1000 samples.</i><br>
Note: percent error does not necessarily reflect how close the heuristic path is to the optimal path.  In this case the heuristic path typically has 0% error however when the path is not perfect, the error is slightly less than the errors reported in the previous table on average.</b>

| method | time complexity | wall time for 20 cities | typical percentage over optimal path length |
|:---:|:---:|:---:|:---:|
|Brute Force| O(n!)| TLE | 0% |
|Dynamic Programming |O(n<sup>2</sup> 2<sup>n</sup>)| &approx; 31 sec | 0% |
|Heuristic Approach | O(n<sup>2</sup>) | &approx; 1 ms | 14%
|Heuristic Approach with Path Relaxation | O(n<sup>2</sup>) | &approx; 3 ms | 5% |
|Heuristic Approach with Path Relaxation and k-subset Optimization | O(n k<sup>2</sup> 2<sup>k</sup>) | &approx; 500 ms | 0%|

<i>\*Typical error is based on a set size of 20 vertices.</i><br>
<i>\*k is floor(n / 2) for the k-subset optimization</i>

### Conclusion

Here we explored 5 approaches to solving the traveling salesman problem.<br>
Two of which are guaranteed to find the optimal solution, at the expense of time and space.

The other three trade off accuracy with speed, ranging from an O(n<sup>2</sup>) solution that on average finds a path 14% longer than the optimal path, to an O(n&middot;k<sup>2</sup>&middot;2<sup>k</sup>) solution that typically finds the optimal path for a set of 20 nodes.  

However which solution is best really depends on your use case.  If you need 100% accuracy, then you have to bite the bullet and use the dynamic programming approach.  And if your problem has thousands of nodes, then you must use some sort of approximation method.  And if you are typically dealing with 15 to 20 nodes, then you can pick an option whose speed and accuracy are ballanced to fit your projects priorities.  

This is an age old problem there are certain to be many approximation methods out there.

To those who are reading this - cheers to you!