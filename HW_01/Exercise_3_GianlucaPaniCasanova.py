#Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

n = 50000

#Draw n random numbers from normal distribution
#def RandNorm(mean):
def RandNorm(l):
    return np.random.normal(loc = l, size = n)
#Values according to exercise
X = RandNorm(0)
Y = RandNorm(2)

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

#Setup variables for Eigen-decomposition and calculations
SampleCov = np.cov(df)
#Split Eigenvalues and -vectors
WMEigenValues = WMEigenDecomposed[0]
WMEigenVectors = WMEigenDecomposed[1] 

#Eigenvoodoo
WMEigenVectors[:,0] = WMEigenVectors[:,0] * np.sqrt(WMEigenValues[0])
WMEigenVectors[:,1] = WMEigenVectors[:,1] * np.sqrt(WMEigenValues[1])

#Calculations according to last Slide of second lecture
CorrectedValues = np.dot(WMEigenVectors,D)

#Print the final covariance matrix and sample means
print("covariance matrix: ", np.cov(CorrectedValues))
print("mean X: ", np.mean(X))
print("mean Y: ", np.mean(Y))
#print(np.cov(df))
#Inspecting Eigenvalues
#m = np.cov(df)
#eig = np.linalg.eig(m)
#print(eig[0])
#print(eig[1])

