#Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

n = 50000

#Draw 1000 random numbers from normal distribution
#def RandNorm(mean, sd):
#    return np.random.normal(loc = mean,scale = sd,size = n)
def RandNorm():
    return np.random.normal(size = n)
#Values according to exercise
X = RandNorm()
Y = RandNorm()

#Create 500x2 matrix
D = np.array([X, Y])
np.transpose(D)

#Inspect results
df = pd.DataFrame(D)

#cov() should output something similar to
#[[ 9   -5.4]
# [-5.4  4  ]]

#Create wanted covariance matrix
WantedMatrix = np.array([[9,-5.4], [-5.4, 4]])

#Eigendecompose WantedMatrix
WMEigenDecomposed = np.linalg.eigh(WantedMatrix)

#Correct sample covariance
SampleCov = np.cov(df)
WMEigenValues = WMEigenDecomposed[0]
WMEigenVectors = WMEigenDecomposed[1] 
print(WMEigenValues)
WMEigenVectors[:,0] = WMEigenVectors[:,0] * np.sqrt(WMEigenValues[0])

WMEigenVectors[:,1] = WMEigenVectors[:,1] * np.sqrt(WMEigenValues[1])

CorrectedValues = np.dot(WMEigenVectors,D)

print(np.cov(CorrectedValues))
#print(np.cov(df))
#Inspecting Eigenvalues
#m = np.cov(df)
#eig = np.linalg.eig(m)
#print(eig[0])
#print(eig[1])

