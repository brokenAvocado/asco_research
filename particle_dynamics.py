import numpy as np
import matplotlib.pyplot as plt
import math

x = np.array([0.,0.,1.,2.,0.])
tf = 5.0  #final_time
dt = .02
xhist = np.array([x])

def particle_dynamics(x, u):
    p1 = x[0] #current x pos
    p2 = x[1] #current y pos
    v1 = x[2] #current x velo
    v2 = x[3] #current y velo
    a1 = u[0] #current x accel
    a2 = u[1] #current y accel
    return np.array([v1, v2, a1, a2, 1.0]) #xdot or resultant

while x[4] < tf:
    u = np.array([x[4]**2-x[4]+1,x[4]**2-x[4]+1])
    x += particle_dynamics(x, u)*dt
    xhist = np.append(xhist, np.array([x]), axis=0)

plt.figure()
plt.plot(xhist[:, 0], xhist[:, 1])
plt.figure()
plt.plot(xhist[:, 4], (np.sqrt(xhist[:, 2]**2 + xhist[:, 3]**2)))
plt.show()
