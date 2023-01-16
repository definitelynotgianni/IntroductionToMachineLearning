import numpy as np
from matplotlib import pyplot as plt
np.random.seed(42)
n = 500

#function for drawing s random numbers from normal distribution
def RandNorm( s ):
    return np.random.normal(loc = 3, scale = np.sqrt(3), size = s)

#create vectors
X = RandNorm(n)
W = RandNorm(n)
Y = RandNorm(n)
Z = RandNorm(n)

D = np.array([X,W,Y,Z])
D = np.transpose(D)

#define functions
def normalize(data):
    return (( data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)))

def scale(data):
    return (( data - (( np.min(data, axis=0) + np.max(data, axis=0)) / 2 )) / (( np.max(data, axis=0) - np.min(data, axis=0)) / 2 ))

def standardize(data):
    return ( (data - np.mean(data, axis=0)) / np.std(data, axis=0) )


print(normalize(D))
print(scale(D))
print(standardize(D))

