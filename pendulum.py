#!/usr/bin/env python
# 2017-04-05
# Yeonjung Hong, Yejin Cho, Youngsun Cho
# The 2nd assignment

import numpy as np
from scipy import integrate
from scipy.signal import argrelextrema
import matplotlib.pylab as plt


# Define the differential equation using Euler's theorem
def pODE(y,t,a,b):
    # alpha: a func of the angle of deflection
    # beta: the angular speed of the pendulum = the time-derivative of the angle, alpha
    # a: a constant related to earth gravity
    # b: a constant related to friction
    alpha, beta = y
    dalpha_dt = beta
    dbeta_dt = - b*beta - a*np.sin(alpha)
    return dalpha_dt, dbeta_dt

# Default assumption for gravity constant a & friction constant b
a = 5
b = 0.25

# Posit a pendulum which is almost extended fully upward and has initial velocity of 0.
alpha0 = np.pi-0.01 # Initial angle
beta0 = 0 # Initial velocity

# Combine into vector
x0 = alpha0, beta0

# time-points to calculate ranging from 0 to 50 (the more time points, the smoother the graph)
t = np.linspace(0, 50, 500)

# Solve the ODE and save solutions
r = integrate.odeint(pODE, x0, t, args=(a, b)) # Put ODE and its arguments
alpha, beta = r.T

# -------------------------------------------------------
# plot the solutions in one plot
fig1, ax = plt.subplots()
fig1.suptitle('Damped pendulum movement', fontsize=14, fontweight='bold')
ax.plot(t,alpha,'aquamarine', lw=2, label='angle')
ax.plot(t,beta,'plum', lw=2, label='angular velocity')
ax.set_xlabel('Time')
ax.set_ylabel('Angle and Angular Velocity')
ax.grid()

max_beta_idx = argrelextrema(beta, np.greater) # Get indices of maximal angular velocities (including local maxima)
max_beta = beta[max_beta_idx] # Maximal angular velocities
max_beta_t = t[max_beta_idx] # Corresponding time points
max_beta_alpha = alpha[max_beta_idx] # Corresponding angles

ax.plot(max_beta_t, max_beta, 'o', color='red',alpha=0.5, ms=10, label='max angular speed') # plot the maximal angular speed
ax.plot(max_beta_t, max_beta_alpha, 'o', color='k', alpha=0.5, ms=5,label='angle for max speed') # plot the corresponding angle
ax.annotate('Speed:{0:.2f}\nAngle:{2:.2f}\nTime:{1:.2f}'.format(max_beta[0], max_beta_t[0],max_beta_alpha[0] ),
            xy=(max_beta_t[0], max_beta[0]),  xycoords='data',
            xytext=(0.3, 0.85), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.5', color='grey'),
            horizontalalignment='left', verticalalignment='top',
            )
ax.text(0.2, 0.03, '''The angular velocity is maximal \nwhen the gradient of angle is the highest by definition, \nand that is where the angles are near zero. ''',
        transform = ax.transAxes, fontweight='bold'
        )

plt.legend(loc='best')

# save the figure
fig1.savefig("pendulum_1.png")
plt.show(block=False)

# -------------------------------------------------------
# [[QUESTION 1]]
# When is the angular velocity is maximal?
# [[ANSWER 1]]
# The angular velocity is maximal when the gradient of angle is the highest by definition.
# And that is where the angles are near ZERO.

# -------------------------------------------------------
# Quiver plot
values = np.linspace(0,1.0,8)

# colors for each trajectory
vcolors = plt.cm.autumn_r(np.linspace(0.3, 1., len(values)))
fig2 = plt.figure()

# plot trajectories
for v, col in zip(values, vcolors):
    alpha0 = np.pi - 0.01
    beta0 = 10
    base = alpha0,beta0
    # starting point
    x0 = [ v * x for x in base ]
    # integrate the ODE for the times and starting points
    r = integrate.odeint(pODE, x0, t, args=(a,b))
    alpha, beta = r.T
    # plot the trajectory with varying linewidth and color
    plt.plot(alpha,beta, lw=3.5 * v, color=col, label='x0 = ({0:.2f}, {1:.2f})'.format(x0[0],x0[1]))
    plt.legend(loc='best')

# -------------------------------------------------------
# define a grid and compute direction at each point
# get axis limits
x_min=-30; y_min=-10
y_max = plt.ylim(ymin=y_min)[1]
x_max = plt.xlim(xmin=x_min)[1]
nb_points = 30

x = np.linspace(x_min, x_max, nb_points)
y = np.linspace(y_min, y_max, nb_points)
# create a grid with the axis limits
X1, Y1 = np.meshgrid(x, y)
# compute growth rate on the grid!
DX1, DY1 = pODE([X1, Y1],t,a,b)
C = np.sqrt(DX1**2 + DY1**2)
# -------------------------------------------------------
# Draw direction fields, using matplotlib's quiver-function
# Arrows are plotted in the same size, but colors are used
# to give information about the growth speed
plt.title('Trajectories and direction fields')
Q = plt.quiver(X1, Y1, DX1, DY1, C, pivot='mid', cmap=plt.cm.jet)
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.legend()
plt.grid()

# save the figure
fig2.savefig('pendulum_2.png')

plt.show()


# [[QUESTION 2]]
# Qualitatively describe the motion of the pendulum
# for the fastest versus the second-slowest trajectory.
# What does the pendulum do?

# [[ANSWER 2]]
# While the pendulum in the second-slowest trajectory, which has very small initial velocity, keeps swinging back and forth around (0,0),
# the fastest trajectory of which initial velocity is the largest initially fluctuates from the top to the lower positions,
# but then due to the effect of friction, it ultimately behaves so much similar to the second-slowest trajectory.

