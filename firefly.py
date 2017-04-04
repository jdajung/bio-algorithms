import random
import util
import test_funs
import sys
import numpy as np
import math
from math import pow

#An implementation of the Firefly Algorithm as found in Yang's MATLAB code
def firefly_matlab(constParams, tuneParams):
    evalFun = constParams[0]
    evalConstParams = constParams[1]
    paramRanges = constParams[2]
    stopCond = constParams[3]
    stopVal = constParams[4]
    repetitions = constParams[5]
    verbose = constParams[6]
    numFlies = tuneParams[0]
    alpha = tuneParams[1]
    betaMin = tuneParams[2]
    gamma = tuneParams[3]

    numDim = len(paramRanges)
    posns = [util.uniform_random(paramRanges) for i in range(numFlies)]
    bestSol = None
    bestSolFitness = sys.float_info.max
    bestSolReps = None
    stopLoop = False
    numEvals = 0
    iterations = 0

    while not stopLoop:
        #Discount alpha
        if stopCond == "generations":
            alpha = alpha_reduce(alpha, stopVal[0])
        else:
            alpha = alpha_reduce(alpha, 100)
        
        #Note that, under this implementation, the best solution is always one iteration behind the movement
        bestSol, bestSolFitness, bestSolReps, fitness, fitnessReps, newEvals = evaluate_sols(evalFun, evalConstParams, posns, repetitions)
        numEvals += newEvals        

        posns = firefly_move_matlab(posns, fitness, alpha, betaMin, gamma, paramRanges)
        
        iterations += 1

        if stopCond == "fitness":
            if bestSolFitness < stopVal[0]:
                stopLoop = True
        elif stopCond == "fitnessOrEvals":
            if bestSolFitness < stopVal[0] or numEvals > stopVal[1]:
                stopLoop = True
        elif stopCond == "fitnessOrGlobalEvals":
            if bestSolFitness < stopVal[0] or test_funs.numEvaluations > stopVal[1]:
                stopLoop = True
        elif stopCond == "generations":
            if iterations >= stopVal[0]:
                stopLoop = True 
        else:
            print "ERROR: Stop condition not recognized"
            break

        if verbose:
            print "Iteration: " + str(iterations)
            print "Evaluations: " + str(numEvals)
            print "Best Solution: " + str(bestSol)
            print "Best Mean Fitness: " + str(bestSolFitness)
            print "Best Fitness Set: " + str(bestSolReps) + "\n"

    return (numEvals, bestSol, bestSolFitness, bestSolReps)


#An implementation of the Firefly Algorithm as described in Yang's paper
def firefly_paper(constParams, tuneParams):
    evalFun = constParams[0]
    evalConstParams = constParams[1]
    paramRanges = constParams[2]
    stopCond = constParams[3]
    stopVal = constParams[4]
    repetitions = constParams[5]
    verbose = constParams[6]
    numFlies = tuneParams[0]
    alpha = tuneParams[1]
    beta0 = tuneParams[2]
    gamma = tuneParams[3]

    numDim = len(paramRanges)
    posns = [util.uniform_random(paramRanges) for i in range(numFlies)]
    bestSol = None
    bestSolFitness = sys.float_info.max
    bestSolReps = None
    stopLoop = False
    numEvals = 0
    iterations = 0

    while not stopLoop:

        #Note that, under this implementation, the best solution is always one iteration behind the movement
        bestSol, bestSolFitness, bestSolReps, fitness, fitnessReps, newEvals = evaluate_sols(evalFun, evalConstParams, posns, repetitions)
        numEvals += newEvals        

        posns = firefly_move_paper(posns, fitness, alpha, beta0, gamma, paramRanges)
        
        iterations += 1

        if stopCond == "fitness":
            if bestSolFitness < stopVal[0]:
                stopLoop = True
        elif stopCond == "fitnessOrEvals":
            if bestSolFitness < stopVal[0] or numEvals > stopVal[1]:
                stopLoop = True
        elif stopCond == "fitnessOrGlobalEvals":
            if bestSolFitness < stopVal[0] or test_funs.numEvaluations > stopVal[1]:
                stopLoop = True
        elif stopCond == "generations":
            if iterations >= stopVal[0]:
                stopLoop = True 
        else:
            print "ERROR: Stop condition not recognized"
            break

        if verbose:
            print "Iteration: " + str(iterations)
            print "Evaluations: " + str(numEvals)
            print "Best Solution: " + str(bestSol)
            print "Best Mean Fitness: " + str(bestSolFitness)
            print "Best Fitness Set: " + str(bestSolReps) + "\n"

    return (numEvals, bestSol, bestSolFitness, bestSolReps)


#Apply a discounting factor to alpha over time
#Must know the expected number of generations
#The other values here seem to have been hand-selected
def alpha_reduce(alpha, generations):
    delta = 1 - pow(pow(10,-4)/0.9, 1.0/generations)
    return alpha * (1-delta)


#Evaluate the fitness of all firefly positions and determine which is the best
def evaluate_sols(evalFun, evalConstParams, posns, repetitions):
    numEvals = 0
    returnFitness = []
    returnReps = []
    bestSolFitness = sys.float_info.max
    bestSol = None
    bestSolReps = None
    for i in range(len(posns)):
        fitnessReps = []
        currSol = posns[i]
        for j in range(repetitions):
            currFitness,_,_,_ = evalFun(evalConstParams, currSol)
            numEvals += 1
            fitnessReps.append(currFitness)
        newFitness = np.mean(fitnessReps)
        returnFitness.append(newFitness)
        returnReps.append(fitnessReps)

        #Check if global bests need to be changed
        if newFitness < bestSolFitness:
            bestSolFitness = newFitness
            bestSol = currSol
            bestSolReps = fitnessReps
    return (bestSol, bestSolFitness, bestSolReps, returnFitness, returnReps, numEvals)


#Perform firefly movements, as done in the MATLAB code
def firefly_move_matlab(posns, fitness, alpha, betaMin, gamma, paramRanges):
    newPosns = [posns[i][:] for i in range(len(posns))]  
    scale = [abs(paramRanges[i][-1]-paramRanges[i][0]) for i in range(len(paramRanges))]
    beta0 = 1
    numFlies = len(posns)
    numDim = len(paramRanges)

    for i in range(numFlies):
        for j in range(numFlies):
            if fitness[i] > fitness[j]: #not i==j unnecessary
                r = util.eucl_dist(posns[i], posns[j])
                beta = (beta0-betaMin)*math.exp(-gamma*pow(r,2))+betaMin #Note that this line is different to the paper
                randFactor = [alpha*(random.random()-0.5)*scale[k] for k in range(numDim)]
                newPosns[i] = [newPosns[i][k]*(1-beta)+posns[j][k]*beta + randFactor[k] for k in range(numDim)]
                newPosns[i] = util.boundary_clamp(newPosns[i], paramRanges)

    return newPosns


#Perform firefly movements, as described in the paper
def firefly_move_paper(posns, fitness, alpha, beta0, gamma, paramRanges):
    newPosns = [posns[i][:] for i in range(len(posns))]  
    scale = [abs(paramRanges[i][-1]-paramRanges[i][0]) for i in range(len(paramRanges))]
    numFlies = len(posns)
    numDim = len(paramRanges)

    for i in range(numFlies):
        for j in range(numFlies):
            if fitness[i] > fitness[j]: #not i==j unnecessary
                r = util.eucl_dist(posns[i], posns[j])
                beta = beta0*math.exp(-gamma*pow(r,2))
                randFactor = [alpha*(random.random()-0.5)*scale[k] for k in range(numDim)]
                newPosns[i] = [newPosns[i][k]*(1-beta)+posns[j][k]*beta + randFactor[k] for k in range(numDim)]
                newPosns[i] = util.boundary_clamp(newPosns[i], paramRanges)

    return newPosns

