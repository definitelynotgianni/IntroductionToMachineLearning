#Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(s, n, mu,cov):
    #Setup function
    np.random.seed(s)

    #Draw n random numbers from normal distribution
    #def RandNorm(mean):
    def RandNorm(l):
        return np.random.normal(loc = l, size = n)
    #Values according to exercise

    X = RandNorm(mu[0])
    Y = RandNorm(mu[1])

    #Create 500x2 matrix
    D = np.array([X, Y])
    np.transpose(D)

    #Inspect results
    df = pd.DataFrame(D)

    #cov() should output something similar to
    #[[ 9   -5.4]
    # [-5.4  4  ]]

    #Take wanted covariance matrix
    WantedMatrix = cov

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
    np.transpose(D)
    CorrectedValues = np.dot(WMEigenVectors,D)
    
    #Reshape CorrectedValues
    np.transpose(CorrectedValues)
    
    #Create payload that returns everything as an item of a list 
    payload = [CorrectedValues, np.array([np.mean(X),np.mean(Y)]), np.cov(pd.DataFrame(CorrectedValues))]

    #Print the final covariance matrix and sample means
    return payload

#solution = f(42, 1000, [0,2], [[9, -5.4],[-5.4, 4]])
#print("values: ", solution[0])
#print("means: ", solution[1])
#print("cov: ", solution[2])
#print(np.cov(df))
#Inspecting Eigenvalues
#m = np.cov(df)
#eig = np.linalg.eig(m)
#print(eig[0])
#print(eig[1])

