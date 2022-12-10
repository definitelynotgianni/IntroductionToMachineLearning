#Aufgaben
#1. Create X and Y by drawing 500 random pointsfrom a normal distribution (done)
#2. Put them in a 500x2 Matrix called D (done)
#3. Make a 2D plot ()

#Setup
import numpy as np 
np.random.seed(42)

#function for drawing s random numbers from normal distribution
def RandNorm( s ):
    return np.random.normal( size = s )

#create two lists filled with random floats drawn from a normal distribution 
X = RandNorm(500)
Y = RandNorm(500) 

#create arry by putting in the two lists created earlier and transpose it to get the right shape
D = np.array([X,Y])
np.transpose(D)

print(np.shape(D)) #REMOVE BEFORE SUBMITTING! ONLY FOR TEST PURPOSES