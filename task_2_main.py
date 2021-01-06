#-----------------------------------------------------------------------------
#PHY3042
#Assignment 2
#Student: 6463360
#Task 2
#---Programme Description-----------------------------------------------------
#This programme takes discrete data and finds the Fourier coefficients of them.
#Once this is done, a continous function is calculated up to the n=7 (that is
#the sin(7x) and cos(7x)) terms in the Fourier series. The truncated Fourier
#series and origional data points are plotted for comparison.
#-----------------------------------------------------------------------------
#Import useful libraries.
import numpy as np
import matplotlib.pyplot as plt


#-Main------------------------------------------------------------------------
#Define data.
yj = [-0.2, -0.1, 0.3, 0.2, 0.4, 0.5, 0.0, -0.4, -0.4, -0.2, 0.1, 0.2, 0.2, \
        0.1, 0.1, -0.1]
xj = [ i/len(yj) * 2 * np.pi for i in range(len(yj)) ]


#Set number of data points.
n = len(yj)
m = n//2

#Check number of data points is even.
if 2*m != n:
    print("ERROR: Number of points not even.")
    exit(1)


#Calculate the Fourier Coefficients using the Fast Fourier Transform.
ck = np.fft.rfft(yj, n=len(yj), axis=0) / m


#Calculate the a and b coefficients from the FFT.
N = 1000    #1000 is enough to produce a nice continous function quickly.
xp = np.linspace(-2.0 * np.pi, 2.0 * np.pi, N)
yp = np.empty(N)
for i, xval in enumerate(xp):
    yp[i] = ck[0].real
    for k in range(1,m):    #Up to the n=7 term in the series.
        ak = (ck[k] + ck[-k]).real
        bk = (ck[-k] - ck[k]).imag
        yp[i] += ak * np.cos(k * xval) + bk * np.sin(k * xval)
    yp[i] += ck[-m].real * np.cos(m * xval)


#Plot the truncated Fourier sum along with the origional data points.
plt.plot(xj, yj, 'o', label="Origional Data Points")
plt.plot(xp, yp, label="Truncated Fourier Series")

plt.title("Truncated Fourier Transform of Data Points")

plt.xlabel("X", fontsize=12); plt.xlim(xp[0], xp[-1])
plt.ylabel("Y", fontsize=12); plt.ylim(min(yp), max(yp))

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)

plt.legend(prop={'size': 12}, loc=8)

plt.tight_layout()
plt.show()