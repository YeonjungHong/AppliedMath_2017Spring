#!/usr/bin/env python
# 2017-04-05
# Yeonjung Hong, Yejin Cho, Youngsun Cho
# The 2nd assignment

import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from scipy import integrate
from decimal import Decimal


# Define logistic differential equation with default parameters
def logistics_diff(P, t, A, K):
    # P: Population
    # A: growth parameter
    # K: carrying capacity
    dP_dt = A*P*(1-P/K)
    return dP_dt


def onclick(event):
    if isinstance(event.artist, Line2D):
        for txt in fig.texts:
            txt.set_visible(False)

        annotations = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Annotation)]
        for anno in annotations:
            anno.set_visible(False)

        thisline = event.artist
        ind = event.ind
        time = np.take(thisline.get_xdata(), ind)[0]
        data = np.take(thisline.get_ydata(), ind)[0]
        ax.annotate('Current time {0:.1f} population {1:.2e}'.format(time, Decimal(data)), xy=(time, data), xytext=(-20, 20),
                    textcoords='offset points', ha='center', va='bottom',
                    bbox=dict(boxstyle='round, pad=0.2', fc='yellow', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='red'))
        y_earlier = logistics_diff(data, t[::-1], A, K)
        y_later = logistics_diff(data, t, A, K)
        fig.text(0.1, 0.15, 'Evaluation towards Earlier times: time=(%.1f), population=(%.2e)' % (time, Decimal(y_earlier)), fontsize=11)
        fig.text(0.1, 0.10, 'Evaluation towards Later times: time=(%.1f), population=(%.2e)' % (time, Decimal(y_later)), fontsize=11)
        fig.canvas.draw()
        fig.canvas.flush_events()



# Set initial population
P0_10 = 10
P0_1500 =1500

# the total time we want to get our data for
t = np.arange(0, 101)

# growth parameter and carrying capacity
A=.08; K=1000

# get population for every time point given 2 different starting points
P_0 = integrate.odeint(logistics_diff, P0_10, t, args=(A, K))
P_1500 = integrate.odeint(logistics_diff, P0_1500, t, args=(A, K))

# Plot
# Here, we plot the data
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t,P_0,color='navy',marker='.', lw=2, label='Initial point={0}'.format(P0_10), picker=5)
ax.plot(t,P_1500,color='crimson',marker='.', lw=2, label='Initial point={0}'.format(P0_1500), picker=5)
ax.set_ylim(0,1500)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.legend()
ax.set_title('click on points', picker=True)
fig.subplots_adjust(bottom=0.3)
fig.canvas.mpl_connect('pick_event', onclick)

# -------------------------------------------------------
# define a grid and compute direction at each point
# get axis limits
x_min=0; y_min=0
y_max = plt.ylim(ymin=y_min)[1]
x_max = plt.xlim(xmin=x_min)[1]
nb_points = 30

x = np.linspace(x_min, x_max, nb_points)
y = np.linspace(y_min, y_max, nb_points)

# create a grid with the axis limits
X, Y = np.meshgrid(x, y)

# compute growth rate on the grid!
DX = logistics_diff(X, t, A, K)
DY = logistics_diff(Y, t, A, K)
C = np.sqrt(DX**2 + DY**2)
# -------------------------------------------------------
# Draw direction fields, using matplotlib's quiver-function
# Arrows are plotted in the same size, but colors are used
# to give information about the growth speed

plt.title('Trajectories and direction fields')
Q = plt.quiver(X,Y, DX, DY,C, pivot='mid')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()