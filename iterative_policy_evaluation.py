from __future__ import print_function, division
from builtins import range
import numpy as np 
from Grid import standard_grid

SMALL_ENOUGH = 1e-3 #threshold for convergence

def print_policy(P,g):
    for i in range(g.width):
        print("-----------------------")
        for j in range(g.height):
            a = P.get((i,j),' ')
            print(" %s |" %a, end=' '),

        print (' ')

def print_values(V,g):
    for i in range(g.width):
        print("-----------------------------")
        for j in range(g.height):
            v= V.get((i,j),0)
            if v>= 0:
                print(" %.2f|" %v, end=' '),
            else:
                print("%.2f|" %v, end=' '),
        print (' ')



if __name__ == "__main__":
    g = standard_grid()
    states = g.all_states()
    #print(states)
    V = {}
    gamma = 1.0
    
    for s in states:
        V[s] = 0
    Iter=0

    while Iter<1000:
        Iter+=1
        print("Iteration %d" %Iter)
        biggest_change  = 0
        #Backup old policy
        for s in states:
            old_v = V[s]
            #V[s] has value only if its not a terminal state
            if s in g.actions:
                new_v = 0
                p_a = 1.0/len(g.actions[s])

                for a in g.actions[s]:
                    g.set_state(s)
                    r = g.move(a)
                    new_v += p_a*(r + gamma * (V[g.current_state()]))
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v-V[s]))
        if biggest_change<SMALL_ENOUGH:
            break
    print("Values for uniformly random actions:")
    print_values(V,g)
    print("\n\n")

    fixed_policy = {
        (2,0):'U',
        (1,0):'U',
        (0,0):'R',
        (0,1):'R',
        (0,2):'R',
        (1,2):'R',
        (2,1):'R',
        (2,2):'R',
        (2,3):'U',
        }
    print_policy(fixed_policy,g)

    V = {}
    gamma = 0.9
    
    for s in states:
        V[s] = 0
    Iter=0

    while Iter<1000:
        Iter+=1
        print("Iteration %d" %Iter)
        biggest_change  = 0
        #Backup old policy
        for s in states:
            old_v = V[s]
            #V[s] has value only if its not a terminal state
            if s in fixed_policy:
                a = fixed_policy[s]
                g.set_state(s)
                r = g.move(a)
                V[s] = r + gamma * V[g.current_state()]
                biggest_change = max(biggest_change, np.abs(old_v-V[s]))
        if biggest_change<SMALL_ENOUGH:
            break
    print("Values for fixed policy:")
    print_values(V,g)
    print("\n\n")

        
        
    