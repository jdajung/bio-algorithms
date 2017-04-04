import random
import util
import test_funs
import sys
import numpy as np
from math import sin, cos, pow, pi
from scipy.special import gamma
from numpy.random import normal, permutation

#An implementation of Cuckoo Search as found in Yang's MATLAB code
def cuckoo_matlab(constParams, tuneParams):
    evalFun = constParams[0]
    evalConstParams = constParams[1]
    paramRanges = constParams[2]
    stopCond = constParams[3]
    stopVal = constParams[4]
    repetitions = constParams[5]
    verbose = constParams[6]
    numNests = tuneParams[0]
    pa = tuneParams[1]

    numEvals = 0
    nests = [util.uniform_random(paramRanges) for i in range(numNests)]
    fitness = [sys.float_info.max for i in range(numNests)]
    fitnessReps = [[sys.float_info.max for j in range(repetitions)] for i in range(numNests)]
    bestNest, fmin, fminReps, nests, fitness, fitnessReps, newEvals = find_best_nests(evalFun, evalConstParams, nests, nests, fitness, fitnessReps, repetitions)
    numEvals += newEvals
    
    iterations = 0
    stopLoop = False
    while(not stopLoop):
        #Do Levy flights
        newNests = levy_flight(nests, bestNest, paramRanges)
        newBestNest, fnew, fnewReps, nests, fitness, fitnessReps, newEvals = find_best_nests(evalFun, evalConstParams, nests, newNests, fitness, fitnessReps, repetitions)
        numEvals += newEvals
        
        #Randomly move nests closer together (sort of)
        newNests = empty_nests(nests, paramRanges, pa)
        newBestNest, fnew, fnewReps, nests, fitness, fitnessReps, newEvals = find_best_nests(evalFun, evalConstParams, nests, newNests, fitness, fitnessReps, repetitions)
        numEvals += newEvals

        #if repetitions > 1:
        #    print "**********"
            
        #Determine new best solution
        if fnew < fmin:
            fmin = fnew
            bestNest = newBestNest
            fminReps = fnewReps
        
        iterations += 1

        if stopCond == "fitness":
            if fmin < stopVal[0]:
                stopLoop = True
        elif stopCond == "fitnessOrEvals":
            if fmin < stopVal[0] or numEvals > stopVal[1]:
                stopLoop = True
        elif stopCond == "fitnessOrGlobalEvals":
            if fmin < stopVal[0] or test_funs.numEvaluations > stopVal[1]:
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
            print "Best Solution: " + str(bestNest)
            print "Best Mean Fitness: " + str(fmin)
            print "Best Fitness Set: " + str(fminReps) + "\n"

    return (numEvals, bestNest, fmin, fminReps)
    
#An implementation of Cuckoo Search as found in Yang's paper
def cuckoo_paper(constParams, tuneParams):
    evalFun = constParams[0]
    evalConstParams = constParams[1]
    paramRanges = constParams[2]
    stopCond = constParams[3]
    stopVal = constParams[4]
    repetitions = constParams[5]
    verbose = constParams[6]
    numNests = tuneParams[0]
    pa = tuneParams[1]

    numEvals = 0
    nests = [util.uniform_random(paramRanges) for i in range(numNests)]
    fitness = [sys.float_info.max for i in range(numNests)]
    fitnessReps = [[sys.float_info.max for j in range(repetitions)] for i in range(numNests)]
    bestNest, fmin, fminReps, nests, fitness, fitnessReps, newEvals = find_best_nests(evalFun, evalConstParams, nests, nests, fitness, fitnessReps, repetitions)
    numEvals += newEvals
    
    iterations = 0
    stopLoop = False
    while(not stopLoop):
        #Do single Levy flight
        newBestNest, fnew, fnewReps, nests, fitness, fitnessReps, newEvals = levy_flight_paper(nests, bestNest, paramRanges, fmin, fminReps, evalFun, evalConstParams, fitness, fitnessReps, repetitions)
        numEvals += newEvals

        #Determine new best solution
        if fnew < fmin:
            fmin = fnew
            bestNest = newBestNest
            fminReps = fnewReps

        #if repetitions > 1:
        #    print "**********"
        
        #Remove proportion pa of lowest quality nests
        newBestNest, fnew, fnewReps, nests, fitness, fitnessReps, newEvals = empty_nests_paper(nests, pa, bestNest, paramRanges, fmin, fminReps, evalFun, evalConstParams, fitness, fitnessReps, repetitions)
        numEvals += newEvals

        #if repetitions > 1:
        #    print "**********"
            
        #Determine new best solution
        if fnew < fmin:
            fmin = fnew
            bestNest = newBestNest
            fminReps = fnewReps
        
        iterations += 1

        if stopCond == "fitness":
            if fmin < stopVal[0]:
                stopLoop = True
        elif stopCond == "fitnessOrEvals":
            if fmin < stopVal[0] or numEvals > stopVal[1]:
                stopLoop = True
        elif stopCond == "fitnessOrGlobalEvals":
            if fmin < stopVal[0] or test_funs.numEvaluations > stopVal[1]:
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
            print "Best Solution: " + str(bestNest)
            print "Best Mean Fitness: " + str(fmin)
            print "Best Fitness Set: " + str(fminReps) + "\n"

    return (numEvals, bestNest, fmin, fminReps)
 
   
#Evaluate the fitness of all nests and identify the best one
def find_best_nests(evalFun, evalConstParams, oldNests, newNests, oldFitness, oldFitnessReps, repetitions):
    currBest = None
    currBestFitness = sys.float_info.max
    currBestReps = None
    returnNests = []
    returnFitness = []
    returnFitnessReps = []
    newEvals = 0
    for i in range(len(oldNests)):
        fitnessReps = []
        for j in range(repetitions):
            currFitness,_,_,_ = evalFun(evalConstParams, newNests[i])
            newEvals += 1
            fitnessReps.append(currFitness)
        newFitness = np.mean(fitnessReps)
        
        if newFitness < oldFitness[i]:
            returnNests.append(newNests[i])
            returnFitness.append(newFitness)
            returnFitnessReps.append(fitnessReps)
        else:
            returnNests.append(oldNests[i])
            returnFitness.append(oldFitness[i])
            returnFitnessReps.append(oldFitnessReps[i])
        if returnFitness[i] < currBestFitness:
            currBestFitness = returnFitness[i]
            currBest = returnNests[i]
            currBestReps = returnFitnessReps[i]
    return (currBest,currBestFitness,currBestReps,returnNests,returnFitness,returnFitnessReps,newEvals)


#Levy flights as performed in the MATLAB code
def levy_flight(nests, bestNest, paramRanges):
    numNests = len(nests)
    numDim = len(nests[0])

    #Parameters of the Levy distribution. These are not viewed as tunable.
    beta = 3.0/2
    sigma = pow(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*pow(2,((beta-1)/2))),(1/beta))
    lengthscale = [(paramRanges[i][-1] - paramRanges[i][0]) * 0.001 for i in range(len(paramRanges))] #0.01

    for i in range(numNests):
        curr = nests[i]
        u = [normal()*sigma for j in range(numDim)]
        v = [normal() for j in range(numDim)]
        stepFromDist = [u[j]/pow(abs(v[j]), (1.0/beta)) for j in range(numDim)]
        
        #PSO sneaks in here
        stepPSO = [lengthscale[j]*stepFromDist[j]*(curr[j]-bestNest[j]) for j in range(numDim)]
        nests[i] = [nests[i][j] + stepPSO[j]*normal() for j in range(numDim)]
        nests[i] = util.boundary_clamp(nests[i], paramRanges)
      
    return nests


#Levy flights, as described in the paper
def levy_flight_paper(nests, bestNest, paramRanges, fmin, fminReps, evalFun, evalConstParams, oldFitness, oldFitnessReps, repetitions):
    numNests = len(nests)
    numDim = len(nests[0])

    #Parameters of the Levy distribution. These are not viewed as tunable.
    beta = 3.0/2
    sigma = pow(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*pow(2,((beta-1)/2))),(1/beta))
    lengthscale = [(paramRanges[i][-1] - paramRanges[i][0]) * 0.001 for i in range(len(paramRanges))] #0.01

    #Select 2 random nests, one to start the Levy flight from, and one to potentially replace
    randomStart = random.randint(0,numNests-1)
    randomToCompare = random.randint(0,numNests-1)
    startNest = nests[randomStart]

    #Use the same Levy distribution as in the Matlab implementation (Do not sneak in PSO or random from normal distribution)
    u = [normal()*sigma for j in range(numDim)]
    v = [normal() for j in range(numDim)]
    stepFromDist = [u[j]/pow(abs(v[j]), (1.0/beta)) for j in range(numDim)]
    newNest = [startNest[i] + stepFromDist[i]*lengthscale[i] for i in range(numDim)]
    newNest = util.boundary_clamp(newNest, paramRanges)

    #Calculate fitness of new nest
    fitnessReps = []
    newEvals = 0
    for j in range(repetitions):
        currFitness,_,_,_ = evalFun(evalConstParams, newNest)
        newEvals += 1
        fitnessReps.append(currFitness)
    newFitness = np.mean(fitnessReps)

    #Now replace the comparison nest if the new one is better
    returnNests = nests
    returnFitness = oldFitness
    returnFitnessReps = oldFitnessReps
    if newFitness < oldFitness[randomToCompare]:
        returnNests[randomToCompare] = newNest
        returnFitness[randomToCompare] = newFitness
        returnFitnessReps[randomToCompare] = fitnessReps

    #Check if global bests need to be changed
    newBestFitness = fmin
    newBestNest = bestNest
    newBestReps = fminReps
    if returnFitness[randomToCompare] < newBestFitness:
        newBestFitness = returnFitness[randomToCompare]
        newBestNest = returnNests[randomToCompare]
        newBestReps = returnFitnessReps[randomToCompare]
    
    return (newBestNest,newBestFitness,newBestReps,returnNests,returnFitness,returnFitnessReps,newEvals)

#Second inner loop of MATLAB code (after Levy flights)
#This function is really nothing like what is described in the paper
def empty_nests(nests, paramRanges, pa):
    numNests = len(nests)
    numDim = len(nests[0])

    #Interestingly, pa is the probability that a value doesn't change
    #This is the opposite of what the description would suggest, but the metaphor
    #really falls apart here anyway
    K = [[random.random() > pa for j in range(numDim)] for i in range(numNests)]
    
    perm1 = permutation(nests)
    perm2 = permutation(nests)
    randFactor = random.random() #just one call for all values

    #This was probably a mistake in the original Matlab code
    #Instead of moving every solution closer to one other solution at random,
    #this instead takes those movement and vectors and randomly assigns them to solutions
    stepSize = [[randFactor*K[i][j]*(perm1[i][j]-perm2[i][j]) for j in range(numDim)] for i in range(numNests)]
    nests = [util.boundary_clamp([nests[i][j] + stepSize[i][j] for j in range(numDim)], paramRanges) for i in range(numNests)]    
    
    return nests


#Process of dumping bad nests, as described in the paper
def empty_nests_paper(nests, pa, bestNest, paramRanges, fmin, fminReps, evalFun, evalConstParams, oldFitness, oldFitnessReps, repetitions):
    numNests = len(nests)
    numDim = len(nests[0])

    #sort nests by fitness
    sortList = [[nests[i], oldFitness[i], oldFitnessReps[i]] for i in range(numNests)]
    sortList.sort(key=lambda x: x[1])

    #determine the cutoff past which nests will be replaced
    breakpoint = int(round((1-pa)*numNests))
    if breakpoint == numNests:
        breakpoint -= 1
    
    returnNests = []
    returnFitness = []
    returnReps = []
    #Keep nests above the cutoff
    for i in range(breakpoint):
        returnNests.append(sortList[i][0])
        returnFitness.append(sortList[i][1])
        returnReps.append(sortList[i][2])
    
    newEvals = 0
    newBestFitness = fmin
    newBestNest = bestNest
    newBestReps = fminReps
    #Replace nests below cutoff and calculate their fitness
    for i in range(breakpoint, numNests):
        newNest = util.uniform_random(paramRanges)
        newNest = util.boundary_clamp(newNest, paramRanges)

        #Calculate fitness of new nest
        fitnessReps = []
        for j in range(repetitions):
            currFitness,_,_,_ = evalFun(evalConstParams, newNest)
            newEvals += 1
            fitnessReps.append(currFitness)
        newFitness = np.mean(fitnessReps)

        #Add new nest to lists
        returnNests.append(newNest)
        returnFitness.append(newFitness)
        returnReps.append(fitnessReps)

        #Check if global bests need to be changed
        if returnFitness[i] < newBestFitness:
            newBestFitness = returnFitness[i]
            newBestNest = returnNests[i]
            newBestReps = returnReps[i]

    return (newBestNest,newBestFitness,newBestReps,returnNests,returnFitness,returnReps,newEvals)



