import numpy as np

def init():
    global coefficients
    global MinimalFittingLimit
    global UpwardsOn
    global PlotBuffer

    coefficients = {}
    #Channel 1: Axopatch
    coefficients['i1'] = {'a': 0.999, 'E': 0,'S': 5, 'eventlengthLimit': 5e-3}
    #Channel 2: Femto
    coefficients['i2'] = {'a': 0.999, 'E': 0,'S': 5, 'eventlengthLimit': 10e-3}

    MinimalFittingLimit = 1e-3

    UpwardsOn = 0

    PlotBuffer = 250