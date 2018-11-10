# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 09:58:29 2017

@author: hb65402
"""

# ------------------------SOURCE CODE --------------------------
# import the libraries
import random
import math
import numpy as np
LIMIT = 1000

def SA():
    def update_temperature(T, k):
        return T - 0.001
    
    def get_neighbors(i, L):
        assert L > 1 and i >= 0 and i < L
        if i == 0:
            return [1]
        elif i == L - 1:
            return [L - 2]
        else:
            return [i - 1, i + 1]
    
    def make_move(x, A, T):
        # nhbs = get_neighbors(x, len(A))
        # nhb = nhbs[random.choice(range(0, len(nhbs)))]
        nhb = random.choice(range(0, len(list(A)))) # choose from all points
    
        delta = list(A)[nhb] - list(A)[x]
    
        if delta < 0:
            return nhb
        else:
            p = math.exp(-delta / T)
            return nhb if random.random() < p else x
    
    def simulated_annealing(A, SA):
        L = len(a)
        
        x0 = random.choice(range(0, L))
        T = 1.
        k = 1
    
        x = x0
        x_best = x0
    
        while T > 1e-3:
            x = make_move(x, A, T)
            if(A[x] < A[x_best]):
                x_best = x
                SA.append(x/1000.)
                #SA.append(x/1.)
            T = update_temperature(T, k)
            k += 1
    
        print ("iterations:", k)
        return x, x_best, x0
    
    def isminima_local(p, A):
        return all(A[p] < A[i] for i in get_neighbors(p, len((A))))
    
    def func(x):
        val = -x
        #val = -20*np.exp(-0.2*np.sqrt((65.536*x-32.768)*(65.536*x-32.768))/2)-np.exp((np.cos(2*np.pi*(65.536*x-32.768)))/2)+20+np.exp(1)
        if x >=300 and x<=500:
            val = 0.
        if x >=800 and x <= 1000:
            val = 0. 
        if x >=0.3 and x<=0.5:
            val = 0.
        if x >=0.8 and x <= 1.:
            val = 0.                            
        return val
    
    def initialize(L):
        return map(func, range(0, L))
    
    
    A = initialize(LIMIT)
    a = list(A)
    
    local_minima = []
    for i in range(0, LIMIT):
        if(isminima_local(i, a)):
            local_minima.append([i, a[i]])
    
    x = 0
    y = a[x]
    
    for xi, yi in enumerate(A):
        if yi < y:
            x = xi
            y = yi
    global_minumum = x
    
    print ("number of local minima: %d" % (len(local_minima)))
    print ("global minimum @%d = %0.3f" % (global_minumum, a[global_minumum]))
    SA = []
    x, x_best, x0 = simulated_annealing(a,SA)
    print ("Solution is @%d = %0.3f" % (x, -a[x]))
    print ("Best solution is @%d = %0.3f" % (x_best, -a[x_best]))
    print ("Start solution is @%d = %0.3f" % (x0, -a[x0]))
    return SA
