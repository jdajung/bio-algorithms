import math
from math import pow, sqrt, sin, cos, pi, exp, e

global numEvaluations

#All the various test functions (and some extras) are in this file.

#min 0 @(1,1,...,1)
def general_sphere(constParams, tuneParams):
    global numEvaluations
    numEvaluations += 1
    return (sum([pow(val-1, 2) for val in tuneParams]), None, -1, None)

#recommended m=10
def michalewicz(constParams, tuneParams):
    global numEvaluations
    numEvaluations += 1
    m = constParams[0]
    vals = [sin(tuneParams[i]) * pow(sin((i+1)*pow(tuneParams[i],2)/pi),2*m) for i in range(len(tuneParams))]
    return (-sum(vals), None, -1, None)

#min 0 @(0,0), range [-5.12,5.12] for all
def deJong(constParams, tuneParams):
    global numEvaluations
    numEvaluations += 1
    return (sum([pow(val, 2) for val in tuneParams]), None, -1, None)

#min -1 @(pi,pi), range [-100,100] for all 
def easom(constParams, tuneParams):
    global numEvaluations
    numEvaluations += 1
    x = tuneParams[0]
    y = tuneParams[1]
    val = -cos(x)*cos(y)*exp(-pow((x-pi),2)-pow((y-pi),2))
    return (val, None, -1, None)

#min 0 @(0,0,...,0), range [-32.768,32.768] for all
def ackley(constParams, tuneParams):
    global numEvaluations
    numEvaluations += 1
    numDim = len(tuneParams)
    sum1 = sum([pow(tuneParams[i],2) for i in range(numDim)])
    term1 = -20*exp(-0.2*sqrt(sum1/float(numDim)))
    sum2 = sum([cos(2*pi*tuneParams[i]) for i in range(numDim)])
    term2 = -exp(sum2/float(numDim))
    val = term1 + term2 + 20 + e
    return (val, None, -1, None)

#min 0 @(1,1,...,1), range [-2.048,2.048] for all
def rosenbrock(constParams, tuneParams):
    global numEvaluations
    numEvaluations += 1
    val = sum([100.0*pow(tuneParams[i+1]-pow(tuneParams[i],2), 2) + pow(1-tuneParams[i], 2) for i in range(len(tuneParams)-1)])
    return (val, None, -1, None)

