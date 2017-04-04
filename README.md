# A Comparison of Biolgically Inspired Algorithms Under Meta-Optimization

This is the companion code to my project for CS 898 at the University of Waterloo. It contains implementations for Particle Swarm Optimization (PSO), two versions of each of Cuckoo Search and the Firefly Algorithm by Xin-She Yang, and the additional code needed to meta-optimize and test them. To select a meta-optimizer/optimizer/test function combination, edit the part of main.py that says CHANGE THIS PART. To run meta-optimization, type:
python main.py
in a terminal. Output will be printed to that terminal.

If you wish to change/add an optimizer, ensure that it takes two parameters:
constParams: a list of parameters that are constant over a meta-optimization run. The list elements, in order must be:
    evalFun: the test function
    evalConstParams: the list of constParams to be sent to the test function (this is [] for all test functions in test_funs)
    paramRanges: a list of valid parameter ranges for each tunable parameter in the form [[x,y],[w,z],...]
    stopCond: a string represented the stop condtion. Valid strings are: "fitness", "fitnessOrEvals", "fitnessOrGlobalEvals", "generations"
    stopVal: a list of stop values corresponding to the stopCond (e.g. 10 generations is [10])
    repetitions: number of times to repeat evaluations (useful if test function is stochastic)
    verbose: True for verbose output
tuneParams: a list of parameters that a meta-optimizer may optimize. This varies between optimizers, so see individual files.
