from __future__ import print_function, division
from builtins import range
import numpy as np
from Grid import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

all_actions = ['U','D','L','R']
SMALL_ENOUGH=1e-3
gamma = 0.9 

if __name__ == "__main__":
    g = negative_grid()
    print("Rewards: ")
    print_values(g.rewards,g) 

    states = g.all_states()
    V = {}
    policy = {}
    for s in states:
        if s in g.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0

    #print(g.actions.keys())
    #Initialize random policy
    for s in g.actions.keys():
        policy[s] = np.random.choice(all_actions)

    print_policy(policy,g)

    Iter=0
    while True:
        #Iterative Policy Iteration
        while Iter<1000:
            Iter+=1
            print("Iteration %d" %Iter)
            biggest_change  = 0
            #Backup old policy
            for s in states:
                old_v = V[s]
                #V[s] has value only if its not a terminal state
                if s in policy:
                    a = policy[s]
                    g.set_state(s)
                    r = g.move(a)
                    V[s] = r + gamma * V[g.current_state()]
                    biggest_change = max(biggest_change, np.abs(old_v-V[s]))
            if biggest_change<SMALL_ENOUGH:
                break

        #Policy improvement
        is_policy_converged = True

        for s in states:

            if s in policy:

                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                for a in all_actions:
                    g.set_state(s)
                    r = g.move(a)
                    v = r +  gamma * V[g.current_state()]
        #             # print("Reward from state ",s," using move ",a," is: ",r)
                    
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if old_a != new_a:
                    is_policy_converged = False
        if is_policy_converged:
            break

    print_values(V, g)
    print("policy:")
    print_policy(policy, g)