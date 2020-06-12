from __future__ import print_function, division
from builtins import range


import numpy as np 
import matplotlib.pyplot as plt

#Slot machines with win probabilities (known to us, unknown to m-armed bandit)
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
EPSILON = 0.1
NUM_ITER = 10000

class BanditArm:
    def __init__(self,m):
        self.m = m
        self.m_estimate = 0 #Estimated winning probability from current bandit
        self.N = 0. #Number of times current arm was pulled

    def pull(self):
        return np.random.randn() + self.m 

    def update(self,x):
        self.N +=1.
        self.m_estimate = ((self.N - 1)*self.m_estimate + x)/self.N


def run_experiment(m1,m2,m3,eps,N):

    bandits = [BanditArm(m1),BanditArm(m2),BanditArm(m3)]
    EPSILON = eps

    #count number of suboptimal choices
    means = np.array([m1,m2,m3])
    true_best = np.argmax(means)
    count_suboptimal = 0

    data = np.empty(N)
    for Iter in range(N):
        #EPSILON = 1/(Iter+1)
        #EPSILON = 0.001*np.power(0.01,Iter)
        #EPSILON GREEDY ALGORITHM
        if np.random.random()<EPSILON:
            #Pick a Random arm
            j = np.random.choice(len(bandits))

        else:
            #Pick the greedy arm
            j = np.argmax([p.m_estimate for p in bandits])


        x = bandits[j].pull()
        bandits[j].update(x)

        if j!= true_best:
            count_suboptimal += 1

        #for the plot
        data[Iter] = x
    cumulative_average = np.cumsum(data)/(np.arange(N) + 1)

    #Plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)+m1)
    plt.plot(np.ones(N)+m2)
    plt.plot(np.ones(N)+m3)
    plt.xscale('log')
    plt.show

        
    # print mean estimates for each bandit
    for b in bandits:
        print("mean estimate:", b.m_estimate)

    print("Percentage suboptimal for epsilon = %s" %EPSILON, float(count_suboptimal)/N)

    return cumulative_average

    
if __name__ == "__main__":
    m1, m2, m3 = 1.5, 2.5, 3.5
    c_1 = run_experiment(m1, m2, m3, 0.1, 100000)
    c_05 = run_experiment(m1, m2, m3, 0.05, 100000)
    c_01 = run_experiment(m1, m2, m3, 0.01, 100000)

    #log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()
    #linear scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()





