import random
import util
import test_funs
import sys
import numpy as np
from math import pow

#A class for holding particle information
class Particle:
    def __init__(self, posn, velocity):
        self.posn = posn
        self.velocity = velocity
        self.fitness = sys.float_info.max
        self.fitnessReps = None
        self.indBestPosn = posn
        self.indBestFitness = sys.float_info.max
        self.indBestReps = None

#A basic implementation of PSO
def pso_basic(constParams, tuneParams):
    evalFun = constParams[0]
    evalConstParams = constParams[1]
    paramRanges = constParams[2]
    stopCond = constParams[3]
    stopVal = constParams[4]
    repetitions = constParams[5]
    verbose = constParams[6]
    numParticles = tuneParams[0]
    inertia = tuneParams[1]
    selfAdjust = tuneParams[2]
    socialAdjust = tuneParams[3]

    numDim = len(paramRanges)
    initPosns = [util.uniform_random(paramRanges) for i in range(numParticles)]
    paramWidths = [abs(paramRanges[i][-1]-paramRanges[i][0]) for i in range(numDim)]
    velocityRanges = [[-paramWidths[i], paramWidths[i]] for i in range(numDim)]
    initVelocities = [util.uniform_random(velocityRanges) for i in range(numParticles)]
    particles = [Particle(initPosns[i], initVelocities[i]) for i in range(numParticles)]

    bestSol = None
    bestSolFitness = sys.float_info.max
    bestSolReps = None
    stopLoop = False
    numEvals = 0
    iterations = 0
    bestSol, bestSolFitness, bestSolReps, newEvals = evaluate_sols(evalFun, evalConstParams, particles, repetitions, bestSol, bestSolFitness, bestSolReps)
    numEvals += newEvals
    
    while(not stopLoop):
        
        for i in range(numParticles):
            update_particles(particles, bestSol, inertia, selfAdjust, socialAdjust, paramRanges, velocityRanges)

        bestSol, bestSolFitness, bestSolReps, newEvals = evaluate_sols(evalFun, evalConstParams, particles, repetitions, bestSol, bestSolFitness, bestSolReps)
        numEvals += newEvals

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
            print "Iteration: " + str(iterations)
            print "Evaluations: " + str(numEvals)
            print "Best Solution: " + str(bestSol)
            print "Best Mean Fitness: " + str(bestSolFitness)
            print "Best Fitness Set: " + str(bestSolReps) + "\n"

    return (numEvals, bestSol, bestSolFitness, bestSolReps)


#Determine the fitness of all particles
def evaluate_sols(evalFun, evalConstParams, particles, repetitions, currBestSol, currBestSolFitness, currBestSolReps):
    numEvals = 0
    returnFitness = []
    returnReps = []
    bestSolFitness = currBestSolFitness
    bestSol = currBestSol
    bestSolReps = currBestSolReps
    numParticles = len(particles)

    for i in range(numParticles):
        fitnessReps = []
        currParticle = particles[i]
        for j in range(repetitions):
            currFitness,_,_,_ = evalFun(evalConstParams, currParticle.posn)
            numEvals += 1
            fitnessReps.append(currFitness)
        newFitness = np.mean(fitnessReps)
        currParticle.fitness = newFitness
        currParticle.fitnessReps = fitnessReps
        if newFitness < currParticle.indBestFitness:
            currParticle.indBestPosn = currParticle.posn
            currParticle.indBestFitness = newFitness
            currParticle.indBestReps = fitnessReps

        #Check if global bests need to be changed
        if newFitness < bestSolFitness:
            bestSolFitness = newFitness
            bestSol = currParticle.posn
            bestSolReps = fitnessReps
    return (bestSol, bestSolFitness, bestSolReps, numEvals)


#Perform position and velocity changes for all particles
def update_particles(particles, bestSol, inertia, selfAdjust, socialAdjust, paramRanges, velocityRanges):
    numDim = len(bestSol)
    for i in range(len(particles)):
        currParticle = particles[i]
        personalDif = [currParticle.indBestPosn[j] - currParticle.posn[j] for j in range(numDim)]
        globalDif = [bestSol[j] - currParticle.posn[j] for j in range(numDim)]
        currParticle.velocity = [currParticle.velocity[j]*inertia + personalDif[j]*selfAdjust*random.random() + globalDif[j]*socialAdjust*random.random() for j in range(numDim)]
        currParticle.velocity = util.boundary_clamp(currParticle.velocity, velocityRanges)
        currParticle.posn = [currParticle.posn[j] + currParticle.velocity[j] for j in range(numDim)]
        currParticle.posn = util.boundary_clamp(currParticle.posn, paramRanges)

