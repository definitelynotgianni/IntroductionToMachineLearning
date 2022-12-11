#Aufgaben
#1. Create X and Y by drawing 500 random pointsfrom a normal distribution (done)
#2. Create variables W and Z like X and Y (done) 
#3. Put X and Y in a 500x2 Matrix called D (done)
#4. Add W and Z to D as a column (done)
#5. Make two 2D Plots (X and Y) and (W and Z) (done)
#6. Add colors and labels to points(done)

#Setup
import numpy as np 
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
np.random.seed(42)
pointCount = 500

#function for drawing s random numbers from normal distribution
def RandNorm( s ):
    return np.random.normal(loc = 3, scale = np.sqrt(3), size = s)
    

#create two lists filled with random floats drawn from a normal distribution 
X = RandNorm(pointCount)
Y = RandNorm(pointCount) 
W = RandNorm(pointCount)
Z = RandNorm(pointCount)

#create arry by putting in the two lists created earlier and transpose it to get the right shape
D = np.array([X,Y,W,Z])
np.transpose(D)
    
plt.scatter(X, Y, s = 20, c = "orange", label = "X and Y")
plt.scatter(W, Z, s = 20, c = "teal", label = "W and Z")

ax.legend()
plt.show()