#
# Just hold some parameters.
#
# Adapted from original LEEA C# library.
#

class Params:
    POPULATION_SIZE = 20
    MAX_GENERATIONS = 2
    SAMPLE_COUNT = 2                ## 2 examples each evaluation.
    MUTATION_POWER = 0.03           ## Maximum size of mutation
    MUTATION_POWER_DECAY = 0.99     ## Power decayed gradually, 0 disables 1 will leave 0 mutation power at last generation
    MUTATION_RATE = 0.04            ## Proportion of weights to mutate
    MUTATION_RATE_DECAY = 0         ## Decay of mutation rate
    SEX_PROPORTION = 0.5            ## Proportion of offspring produced by sexual reproduction
    SELECTION_PROPORTION = 0.4      ## Top X proportion of individuals selected for reproduction
    INITIAL_WEIGHTS_DELTA = 1       ## Initial weights range from [ -W_D, W_D ]
    FITNESS_DECAY_RATE = 0.2        ## .2 = 20% decay per evaluation
    TRACKING_STRIDE = 1000          ## Every n generations, print out info
    LOW_FITNESS_BETTER = True       ## Is lower fitness better?