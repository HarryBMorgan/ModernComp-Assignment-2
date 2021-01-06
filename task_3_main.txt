#-----------------------------------------------------------------------------
#PHY3042
#Assignment 2
#Student: 6463360
#Task 3
#---Programme Description-----------------------------------------------------
#This programme finds the "volume" of the unit n-ball for multi-dimensional
#space. The dimensions are ranging from 2<=dim<=15. The volume is calculated
#analytically by the function "Vn" for comparing to the Monte Carlo method of
#solving the problem.
#-----------------------------------------------------------------------------
#Import useful libraries.
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt


#-Functions-------------------------------------------------------------------
#Function for analytic solution.
def Vn(n):  #n=number of dimensions.
    return np.pi**(n/2) / gamma(n/2 + 1)


#-Main------------------------------------------------------------------------
#Define variables.
npts = 100000   #Number of points to sample (More npts = larger accuracy).
dim = np.arange(2, 16, 1)   #Dimension cannot be <1.


#Itterate programme for each dimension.
vol, V = [], [] #Holds the 2 volume calculations.
scale = dim[0]  #*vol by fraction of complete vol it is for that dim.
for i, n in enumerate(dim):
    
    #Solve analytically for current dimension.
    V.append(Vn(n))
    
    #Make data list have n arrays of npts of data.
    #Each array holds data for nth dimension.
    data = []
    for j in range(n):  #Populate with random data between 0<=x<=1.
        data.append(np.random.random(npts))
    
    #NOTE to self: data[dimension][npts]
    #Check if point is within unit n-ball.
    count = 0   #Reset for new dimension calculation.
    for k in range(npts):
        val = 0 #Reset for next data point.
        for l in range(n):
            val += data[l][k]**2
        if val < 1.0: count += 1
    scale *= 2  #Scale up for current dimension.
    vol.append(count/npts*scale)
    
    #Progress for user (I just like to know how it's getting on).
    print("Calculating unit n-ball volume for %s dimensions" %n, end='\r')


#Plot the Monte Carlo results against the analytical solution.
plt.plot(dim, vol, color='b', linestyle=':', linewidth=3, \
    label="Monte Carlo Method")
plt.plot(dim, V, color='k', linestyle='-', linewidth=1, \
    label="Analytical Solution")

plt.title("Volume of Unit N-Ball for Varying Dimensions")

plt.xlabel("Dimension", fontsize=12), plt.ylabel("Volume", fontsize=12)
plt.xlim(dim[1], dim[-1])

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)

plt.legend(prop={'size': 12})
plt.tight_layout()  #Makes sure nothing get's cut off when producing image.
plt.show()