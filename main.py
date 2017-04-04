import test_funs
from timeit import Timer
from decimal import Decimal
from cuckoo import cuckoo_matlab, cuckoo_paper
from grid import grid_binary
from random_search import uniform_random_search
from firefly import firefly_matlab, firefly_paper
from pso import pso_basic
from math import pi
import numpy as np

#Some functions for formatting output strings
def numToStr(num, prec):
    returnString = ""
    if num > 100 or (abs(num) < 0.1 and abs(num) > 0):
        returnString = "\\num{" + ("{:." + str(prec) + "E}").format(Decimal(str(num))) + "}"
    else:
        returnString = ("{:." + str(prec) + "f}").format(round(num,prec))
    return returnString

def numToDecStr(num, prec):
    returnString = ("{:." + str(prec) + "f}").format(round(num,prec))
    return returnString

#Define constants needed for tests
test_funs.numEvaluations = 0
coarseTol = 0.01 #Coarse tolerance value used in meta-optimization
tol = 0.00001 #Fine tolerance useed in repeated optimization runs

#Parameter ranges for evaluation functions
paramRanges_5To5_16dim = [[-5.0,5.0] for i in range(16)]
paramRanges_0ToPi_10dim = [[0.0,pi] for i in range(10)]
paramRanges_ackley128dim = [[-32.768,32.768] for i in range(128)]
paramRanges_ackley16dim = [[-32.768,32.768] for i in range(16)]
paramRanges_rosenbrock16dim = [[-2.048,2.048] for i in range(4)]
paramRanges_deJong16dim = [[-5.12,5.12] for i in range(16)]

#Parameters needed for meta-optimization
easomConstParams = [test_funs.easom, [], [[-100.0,100.0],[-100.0,100.0]], "fitnessOrEvals", [-1.0+coarseTol, 20000], 1, False]
deJong16DimConstParams = [test_funs.deJong, [], paramRanges_deJong16dim, "fitnessOrEvals", [0.0+coarseTol, 20000], 1, False]
rosenbrock16DimConstParams = [test_funs.rosenbrock, [], paramRanges_rosenbrock16dim, "fitnessOrEvals", [0.0+coarseTol, 20000], 1, False]
ackley16DimConstParams = [test_funs.ackley, [], paramRanges_ackley16dim, "fitnessOrEvals", [0.0+coarseTol, 20000], 1, False]


#******CHANGE THIS PART*******
metaOpt = grid_binary #grid_binary, uniform_random_search, (pso_basic,...)
opt = pso_basic #pso_basic, cuckoo_matlab, cuckoo_paper, firefly_matlab, firefly_paper
testFunParams = easomConstParams #easomConstParams, deJong16DimConstParams, rosenbrock16DimConstParams, ackley16DimConstParams
#*****************************

metaOptConsts = []
optParamRanges = []

#Meta-optimization parameters
if metaOpt == pso_basic:
    metaOptConsts = [25, 0.6, 1.49, 1.49]
elif metaOpt == cuckoo_matlab or metaOpt == cuckoo_paper:
    metaOptConsts = [25, 0.25]
elif metaOpt == firefly_matlab or metaOpt == firefly_paper:
    metaOptConsts = [25, 0.25, 0.20, 1]

#Parameter ranges for optimizers
if opt == pso_basic:
    optParamRanges = [[15,50], [0.1,1.1], [0.0,3.0], [0.0,3.0]]
elif opt == cuckoo_matlab or opt == cuckoo_paper:
    optParamRanges = [[15,50],[0.0,1.0]]
elif opt == firefly_matlab or opt == firefly_paper:
    optParamRanges = [[15,50],[0.0,1.0],[0.0,1.0],[0.01,100.0]]

numUpperEvals = -1
bestSol = None
bestSolEvals = -1
bestSolReps = None

#function the runs the meta-optimizer
def metaOptDummy():
    global numUpperEvals, bestSol, bestSolEvals, bestSolReps
    numUpperEvals, bestSol, bestSolEvals, bestSolReps = metaOpt([opt, testFunParams, optParamRanges, "fitnessOrGlobalEvals", [0,1000000], 4, True], metaOptConsts)

print "Beginning meta-optimization...\n"

#Do the meta-optimization
t = Timer(metaOptDummy)
dur = t.timeit(number=1)


#Output string stuff
testFunString = ""
if testFunParams[0] == test_funs.easom:
    testFunString = "Easom's"
elif testFunParams[0] == test_funs.deJong:
    testFunString = "De Jong's"
elif testFunParams[0] == test_funs.rosenbrock:
    testFunString = "Rosenbrock's"
elif testFunParams[0] == test_funs.ackley:
    testFunString = "Ackley's"

optString = ""
if opt == pso_basic:
    optString = "PSO"
elif opt == cuckoo_matlab:
    optString = "CuckooM"
elif opt == cuckoo_paper:
    optString = "CuckooP"
elif opt == firefly_matlab:
    optString = "FireflyM"
elif opt == firefly_paper:
    optString = "FireflyP"
optString += "(" + str(bestSol[0])
for i in range(1,len(bestSol)):
    optString += "," + numToDecStr(bestSol[i],1)
optString += ")"

metaOptString = ""
if metaOpt == grid_binary:
    metaOptString = "Grid"
elif metaOpt == uniform_random_search:
    metaOptString = "Random"
elif metaOpt == pso_basic:
    metaOptString = "PSO"
elif metaOpt == cuckoo_matlab:
    metaOptString = "CuckooM"
elif metaOpt == cuckoo_paper:
    metaOptString = "CuckooP"
elif metaOpt == firefly_matlab:
    metaOptString = "FireflyM"
elif metaOpt == firefly_paper:
    metaOptString = "FireflyP"


numRepetitions = 20
time = [0 for i in range(numRepetitions)]
evals = [0 for i in range(numRepetitions)]
fitness = [None for i in range(numRepetitions)]
failures = 0
failureFitness = []
successEvals = []

#Parameters for repeated optimizer runs
optParams = bestSol
if testFunParams[0] == test_funs.easom:
    testFunParams = [test_funs.easom, [], [[-100.0,100.0],[-100.0,100.0]], "fitnessOrEvals", [-1.0+tol, 100000], 1, False]
elif testFunParams[0] == test_funs.deJong:
    testFunParams = [test_funs.deJong, [], paramRanges_deJong16dim, "fitnessOrEvals", [0.0+tol, 100000], 1, False]
elif testFunParams[0] == test_funs.rosenbrock:
    testFunParams = [test_funs.rosenbrock, [], paramRanges_rosenbrock16dim, "fitnessOrEvals", [0.0+tol, 100000], 1, False]
elif testFunParams[0] == test_funs.ackley:
    testFunParams = [test_funs.ackley, [], paramRanges_ackley16dim, "fitnessOrEvals", [0.0+tol, 100000], 1, False]

print "Beginning Repeated Runs...\n"

#Function that runs a single optimizer run
def repeatRunsDummy(i):
    global evals, fitness
    evals[i], _, fitness[i], _ = opt(testFunParams, optParams)

#Do the runs
for i in range(numRepetitions):
    t = Timer(lambda: repeatRunsDummy(i))
    time[i] = t.timeit(number=1)
    test_funs.numEvaluations = 0
    
    #We really want to report the difference from the optimal fitness, not the absolute fitness
    #Easom's is the only one where the optimal is -1 instead of 0
    if testFunParams[0] == test_funs.easom:
        fitness[i] += 1.0

    if evals[i] >= 100000:
        failures += 1
        failureFitness.append(fitness[i])
    else:
        successEvals.append(evals[i])

#More output string stuff
successString = str(int(100*float(numRepetitions-failures)/numRepetitions))

successEvalsString = ""
if failures != numRepetitions:
    successEvalsString = numToDecStr(np.mean(successEvals),1) + "$\\pm$" + numToDecStr(np.std(successEvals),1)
else:
    successEvalsString = "N/A"

failFitnessString = ""
if failures != 0:
    failFitnessString = numToStr(np.mean(failureFitness),2) + "$\\pm$" + numToStr(np.std(failureFitness),2)
else:
    failFitnessString = "N/A"

timeString = numToDecStr(np.mean(time),3)


print "**META-OPTIMIZATION**"
print "Optimizer: " + optString
print "Meta-Optimizer: " + metaOptString
print "Duration: " + str(dur)
print "Total Evaluations: " + str(test_funs.numEvaluations)
print "Optimizer Runs: " + str(numUpperEvals)
print "Best Optimizer Params: " + str(bestSol)
print "Best Optimizer Evals: " + str(bestSolEvals)
print "Best Optimizer Reps: " + str(bestSolReps) + "\n"

print "**OPTIMIZER RUNS**"
print "Evals: " + str(evals)
print "Times: " + str(time)
print "Fitness: " + str(fitness)
print "Failures: " + str(failures) + "\n"

print testFunString + " & " + optString + " & " + metaOptString + " & " + successString + " & " + successEvalsString + " & " + failFitnessString + " & " + timeString + " \\\\"

if failures != numRepetitions:
    print str(np.mean(successEvals))
else:
    print str(100000)

if failures != 0:
    print str(1.0/np.mean(failureFitness))
else:
    print str(1.0/0.00001)

