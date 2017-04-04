import random
import util
import test_funs
import sys
import numpy as np
from math import pow

#An implementation of Grid Search that expands its pattern in a binary fashion
def grid_binary(constParams, tuneParams):
    evalFun = constParams[0]
    evalConstParams = constParams[1]
    paramRanges = constParams[2]
    stopCond = constParams[3]
    stopVal = constParams[4]
    repetitions = constParams[5]
    verbose = constParams[6]
    
    bestSol = None
    bestSolFitness = sys.float_info.max
    bestSolReps = None
    stopLoop = False
    numDim = len(paramRanges)

    numEvals = 0
    iterations = 0
    depth = 0
    pointsPerDim = int(pow(2,depth))
    pointNum = 0
    while(not stopLoop):
        workingPointNum = pointNum
        indices = []
        for i in range(numDim):
            indices.append(workingPointNum % pointsPerDim)
            workingPointNum /= pointsPerDim

        newSol = []
        for i in range(numDim):
            newSol.append(get_val_for_index(indices[i], pointsPerDim, paramRanges[i]))

        #Determine fitness of new solution
        fitnessReps = []
        for j in range(repetitions):
            currFitness,_,_,_ = evalFun(evalConstParams, newSol)
            numEvals += 1
            fitnessReps.append(currFitness)
        newFitness = np.mean(fitnessReps)

        #Check if global bests need to be changed
        if newFitness < bestSolFitness:
            bestSolFitness = newFitness
            bestSol = newSol
            bestSolReps = fitnessReps

        pointNum += 1
        if pointNum >= pow(pointsPerDim, numDim):
            if verbose:
                print "Depth: " + str(depth)
                print "Cumulative Evaluations: " + str(numEvals)
                print "Best Solution: " + str(bestSol)
                print "Best Mean Fitness: " + str(bestSolFitness)
                print "Best Fitness Set: " + str(bestSolReps) + "\n"

                pointNum = 0
                depth += 1
                pointsPerDim = int(pow(2,depth))
        
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
            if float(iterations)/25.0 >= stopVal[0]:
                stopLoop = True 
        else:
            print "ERROR: Stop condition not recognized"
            break

    if verbose:
        print "Iterations: " + str(iterations)
        print "Evaluations: " + str(numEvals)
        print "Best Solution: " + str(bestSol)
        print "Best Mean Fitness: " + str(bestSolFitness)
        print "Best Fitness Set: " + str(bestSolReps) + "\n"

    return (numEvals, bestSol, bestSolFitness, bestSolReps)


#Given a cell's index, return its actual position in parameter space
def get_val_for_index(index, pointsPerDim, paramRange):
    length = paramRange[-1] - paramRange[0]
    chunkLength = float(length)/(pointsPerDim+1)
    returnVal = chunkLength*(index+1) + paramRange[0]
    if isinstance(paramRange[0], int) and isinstance(paramRange[-1], int):
        returnVal = int(round(returnVal))
    return returnVal
    
