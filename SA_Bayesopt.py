# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:42:29 2017

@author: hb65402
"""

from __future__ import print_function
from __future__ import division

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
import numpy as np

from bayes_opt import BayesianOptimization
from SA import SA
# Load data set and target values
data, target = make_classification(
    n_samples=1000,
    n_features=45,
    n_informative=12,
    n_redundant=7
)

def svccv(C):
    val = C
    #val = 20*np.exp(-0.2*np.sqrt((65.536*C-32.768)*(65.536*C-32.768))/2)-np.exp((np.cos(2*np.pi*(65.536*C-32.768)))/2)+20+np.exp(1)
    if C >=0.3 and C<=0.5:
        val = 0.
    if C >= 0.8 and C <= 1:
        val = 0.                      
    return val

def rfccv(n_estimators, min_samples_split, max_features):
    val = cross_val_score(
        RFC(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            random_state=2
        ),
        data, target, 'f1', cv=2
    ).mean()
    return val

if __name__ == "__main__":
    gp_params = {"alpha": 1e-5}
    SA = SA()
    svcBO = BayesianOptimization(svccv,
        {'C': (0., 1.)})
    svcBO.explore({'C': SA})
    #svcBO.explore({'C':[0.1,0.2,0.5,0.9]})                            

                                 
    svcBO.maximize(n_iter=5, **gp_params)
    print('-' * 53)
   
                                 
    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    print('SVC: %s' % list(svcBO.res['max']['max_params'].values())[0])
   

