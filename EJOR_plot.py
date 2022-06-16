# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:52:16 2022

@author: priyank
"""

import matplotlib.pyplot as plt
import pickle


pickling_on = open("EJOR_convex_relaxed","rb")

cum_regret_mc=pickle.load(pickling_on)


a=np.sum(cum_regret_mc,axis=0)

plt.plot(a)