import random
import math
from math import pow

#Produce a potential solution with values in the parameter space selected from a uniform random distribution
def uniform_random(paramRanges):
    vec = []
    for i in range(len(paramRanges)):
        lower = paramRanges[i][0]
        upper = paramRanges[i][-1]
        randVal = random.random()*(upper-lower)+lower
        if isinstance(paramRanges[i][0], int) and isinstance(paramRanges[i][-1], int):
            randVal = int(round(randVal))
        vec.append(randVal)
    return vec

#Prevent parameter values from being outside of allowed ranges
def boundary_clamp(vals, paramRanges):
    for i in range(len(vals)):
        if vals[i] < paramRanges[i][0]:
            vals[i] = paramRanges[i][0]
        elif vals[i] > paramRanges[i][-1]:
            vals[i] = paramRanges[i][-1]
        if isinstance(paramRanges[i][0], int) and isinstance(paramRanges[i][-1], int):
            vals[i] = int(round(vals[i]))
    return vals

#find the Euclidean distance between vec1 and vec2, which must have the same length
def eucl_dist(vec1, vec2):
    sum = 0.0
    for i in range(len(vec1)):
        sum += pow(vec1[i]-vec2[i], 2)
    return math.sqrt(sum)
