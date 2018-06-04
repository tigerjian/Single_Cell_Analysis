# -*- coding: utf-8 -*-

import numpy as np


trans = np.array([[ 1.00536974e+00,-1.02083674e-02,9.09809222e-01], [ 5.41673955e-04,9.97852277e-01,8.31402219e-01], [0,0,1]])

p_1 = np.array([95073.82, 24627.93,1])
p_2 = np.array([104750.15, 31983.67,1])
p_3 = np.array([101250.22, 18094.48,1])
p_4 = np.array([96295.68, 39023.82,1])

print(np.dot(trans,p_1))
print(np.dot(trans,p_2))
print(np.dot(trans,p_3))
print(np.dot(trans,p_4))




