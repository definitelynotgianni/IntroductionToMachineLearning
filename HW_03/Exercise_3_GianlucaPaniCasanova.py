import numpy as np
from matplotlib import pyplot as plt


np.random.seed(42)
n = 50
sample = np.random.normal(size=n)


def normalize(data):
    return (( data - np.min(data)) / (np.max(data) - np.min(data)))

def scale(data):
    return (( data - (( np.min(data) + np.max(data)) / 2 )) / (( np.max(data) - np.min(data)) / 2 ))

def standardize(data):
    return ( (data - np.mean(data)) / np.std(data) )


print(normalize(sample))