#!/usr/bin/env python
'''
From https://github.com/WeatherGod/NNforZR
'''

import numpy			# for std(), sum(), log(), exp() and numpy arrays
from scipy import optimize	# for fmin()

def boxcox_opt(lamb, *pargs):
#   Don't call this function, it is meant to be
#   used by boxcox_auto().
    x = numpy.array(pargs)
    
    # Transform data using a particular lambda.
    xhat = boxcox(x, lamb)
    
    # The algorithm calls for maximizing the LLF; however, since we have
    # only functions that minimize, the LLF is negated so that we can 
    # minimize the function instead of maximixing it to find the optimum lambda.
    return(-(-(len(x)/2.0) * numpy.log(numpy.std(xhat.T)**2) + (lamb - 1.0)*(numpy.sum(numpy.log(x)))))


def boxcox_auto(x):
#   Automatically determines the lambda needed to perform a boxcox
#   transform of the given vector of data points.  Note that it will
#   also automatically offset the datapoints so that the minimum value
#   is just above 0 to satisfy the criteria for the boxcox transform.
#
#   The object returned by this function contains the transformed data
#   ('bcData'), the lambda ('lmbda'), and the offset used on the data
#   points ('dataOffset').  This object can be fed easily into
#   boxcox_inverse() to retrieve the original data values like so:
#
#   EXAMPLE:
#         >>> bcResults = boxcox_auto(dataVector)
#         >>> print bcResults
#	  {'lmbda': array([ 0.313]), 'bcData': array([ 0.47712018,  1.33916353,  6.66393874, ...,  3.80242394,
#                 3.79166974,  0.47712018]), 'dataOffset': 2.2204460492503131e-16}
#         >>> reconstit = boxcox_inverse(**bcResults)
#         >>> print numpy.mean((dataVector - reconstit) ** 2)
#         5.6965875916e-29

    x = x[numpy.isfinite(x)]

    constOffset = -numpy.min(x) + numpy.finfo(float).eps
    tempX = x + constOffset
    bclambda = optimize.fmin(boxcox_opt, 0.0, args=(tempX), maxiter=2000, disp=0)

    # Generate the transformed data using the optimal lambda.
    return({'bcData': boxcox(tempX, bclambda), 'lmbda': bclambda, 'dataOffset': constOffset})

def boxcox(x, lmbda):
#   boxcox() performs the boxcox transformation upon the data vector 'x',
#   using the supplied lambda value 'lmbda'.
#   Note that this function does not check for minimum value of the data,
#   and it will not correct for values being below 0.
#
#   The function returns a vector the same size of x containing the
#   the transformed values.
    if (lmbda != 0.0) :
        return(((x ** lmbda) - 1) / lmbda)
    else :
        return(numpy.log(x))
    

def boxcox_inverse(bcData, lmbda, dataOffset = 0.0) :
#   Performs the inverse operation of the boxcox transform.
#   Note that one can use the output of boxcox_auto() to easily
#   run boxcox_inverse:
#
#   >>> bcResults = boxcox_auto(data)
#   >>> reconstitData = boxcox_inverse(**bcResults)
#
#   Also can be used directly like so:
#   >>> transData = boxcox(origData, lambdaVal)
#   >>> transData = DoSomeStuff(transData)
#   >>> reconstitData = boxcox_inverse(transData, lambdaVal)
#
    if (lmbda != 0.0) :
        
        a = (bcData * lmbda) + 1
        a[a <= 0] = numpy.finfo(float).eps
        
        return((a ** (1.0/lmbda)) - dataOffset)
    else :
        return(numpy.exp(bcData) - dataOffset)


