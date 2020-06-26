from __future__ import print_function, division
from builtins import range
import numpy as np 
from Grid import windy_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3 #threshold for convergence

def print_values(V, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")



if __name__ == "__main__":
    transition_probs = {}
    rewards = {}
    grid = windy_grid()
    for (s,a), v in grid.probs.items():
        for s2, p in v.items():
            transition_probs[(s,a,s2)] = p
            rewards[(s,a,s2)] = grid.rewards.get(s2,0)


    policy = {
        (2,0) : {'U' : 0.5 , 'R' : 0.5},
        (1,0) : {'U' : 1.0},
        (0,0) : {'R' : 1.0},
        (0,1) : {'R' : 1.0},
        (0,2) : {'R' : 1.0},
        (1,2) : {'R' : 1.0},
        (2,1) : {'R' : 1.0},
        (2,2) : {'R' : 1.0},
        (2,3) : {'U' : 1.0},
        }
    print_policy(policy,grid)

    V = {}
    gamma = 0.9
    
    for s in grid.all_states():
        V[s] = 0
    Iter=0

    while Iter<1000:
        biggest_change  = 0
        #Backup old policy
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        action_prob = policy[s].get(a,0)
                        r = rewards.get((s,a,s2),0)
                        new_v += action_prob * transition_probs.get((s,a,s2),0)* ( r + gamma * V[s2] )
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v-V[s]))
        print("Iteration: ", Iter, " Biggest change: ",biggest_change)
        print_values(V,grid)
        Iter+=1

        if biggest_change < SMALL_ENOUGH:
            break
    print("V:", V)
    print("\n\n")