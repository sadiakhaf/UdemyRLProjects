from __future__ import print_function, division
from builtins import range


import numpy as np 
import matplotlib.pyplot as plt

#Slot machines with win probabilities (known to us, unknown to m-armed bandit)
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
#EPSILON = 0.1
NUM_ITER = 10000

class Bandit:
    def __init__(self,p):
        self.p = p
        self.p_estimate = 0. #Estimated winning probability from current bandit
        self.N = 0. #Number of times current arm was pulled

    def pull(self):
        return np.random.random() < self.p 

    def update(self,x):
        self.N +=1.
        self.p_estimate = ((self.N - 1)*self.p_estimate + x)/self.N


def experiment():

    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    print("bandit prob:",[b.p for b in bandits])
    times_explored = 0
    times_exploited = 0
    times_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    print("optimal j:", optimal_j)
    rewards = np.zeros(NUM_ITER)
    #create bandits
    for Iter in range(NUM_ITER):
        EPSILON = 1/(Iter+1)
        #EPSILON = 0.001*np.power(0.01,Iter)
        #EPSILON GREEDY ALGORITHM
        if np.random.random()<EPSILON:
            #Pick a Random arm
            j = np.random.randint(len(bandits))
            times_explored += 1
        else:
            #Pick the greedy arm
            j = np.argmax([p.p_estimate for p in bandits])
            times_exploited += 1

        if j==optimal_j:
            times_optimal += 1

        x = bandits[j].pull()
        rewards[Iter] = x
        bandits[j].update(x)
        
    # print mean estimates for each bandit
    for b in bandits:
        print("mean estimate:", b.p_estimate)

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_ITER)
    print("num_times_explored:", times_explored)
    print("num_times_exploited:", times_exploited)
    print("num times selected optimal bandit:", times_optimal)
    print("Rewards:", rewards)
    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_ITER) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_ITER)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == "__main__":
  experiment()




