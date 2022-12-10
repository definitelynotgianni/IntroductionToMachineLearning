#Aufgaben
#1. Create X and Y by drawing 500 random pointsfrom a normal distribution (done)
#2. Put them in a 500x2 Matrix called D (done)
#3. Make a 2D plot ()

#Setup
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(42)
pointCount = 500

#function for drawing s random numbers from normal distribution
def RandNorm( s ):
    return np.random.normal( size = s )

#create two lists filled with random floats drawn from a normal distribution 
X = RandNorm(pointCount)
Y = RandNorm(pointCount) 

#create arry by putting in the two lists created earlier and transpose it to get the right shape
D = np.array([X,Y])
np.transpose(D)




for i in np.arange(1, pointCount):
    
    plt.scatter(X[:i], Y[:i], s = 20, c = (np.sqrt(X[:i]**2 + Y[:i]**2 )))
    

plt.show()