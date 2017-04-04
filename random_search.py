import random
import util
import test_funs
import sys
import numpy as np
from math import pow

#An implementation of Random Search
def uniform_random_search(constParams, tuneParams):
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
    while(not stopLoop):
        newSol = util.uniform_random(paramRanges)

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
    
