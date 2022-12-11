#Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

n = 1000

#Draw 1000 random numbers from normal distribution
def RandNorm(mean, sd):
    return np.random.normal(loc = mean,scale = sd,size = 500)

#Values according to exercise
X = RandNorm(0, 3)
Y = RandNorm(2, 2)

#Create 500x2 matrix
D = np.array([X, Y])
np.transpose(D)

#Inspect results
df = pd.DataFrame(D)

#cov() should output something similar to
#[[ 9   -5.4]
# [-5.4  4  ]]

print(np.cov(df))

#Inspecting Eigenvalues
m = np.cov(df)
eig = np.linalg.eig(m)
print(eig[0])
print(eig[1])

