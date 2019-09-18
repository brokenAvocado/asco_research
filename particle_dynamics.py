import numpy as np
import matplotlib.pyplot as plt
import math

x = np.array([0.,0.,1.,2.])
xg = np.array([1., 1., 1., 1.])
R = np.diag([1., 1.])
Q = np.diag([0., 1., 1., 1.])
Qf = np.diag([1., 1., 0., 1.])
t = 0.0  #initial_time
tf = 5.0  #final_time
dt = .02
thist = np.array([t])
xhist = np.array([x])

def particle_dynamics(x, u):
    p1 = x[0] #current x pos
    p2 = x[1] #current y pos
    v1 = x[2] #current x velo
    v2 = x[3] #current y velo
    a1 = u[0] #current x accel
    a2 = u[1] #current y accel
    return np.array([v1, v2, a1, a2]) #xdot or resultant

def quadratic_cost_for(x, u, Q, R, Qf, dt, xg):
    a = 0
    for i in range(0, np.size(u, 0)):
       a += (np.dot(np.dot(np.transpose(u[i]),R), u[i]) + np.dot(np.dot(np.transpose(x[i]-xg), Q), (x[i]-xg)))*dt
       a += np.dot(np.dot(np.transpose(x[-1]-xg), Qf), (x[-1]-xg))
    return a

u = np.array([t**2-t+1, t**2-t+1])
uhist = np.array([u])
t += dt
while t < tf:
    x += particle_dynamics(x, u)*dt
    t += dt
    thist = np.append(thist, np.array([t]), axis=0)
    xhist = np.append(xhist, np.array([x]), axis=0)
    u = np.array([t**2-t+1, t**2-t+1])
    uhist = np.append(uhist, np.array([u]), axis=0)

#print(np.size(xhist)) check for size of array (1008!?)
print(quadratic_cost_for(xhist, uhist, Q, R, Qf, dt, xg))
plt.figure()
plt.plot(xhist[:, 0], xhist[:, 1])
plt.figure()
plt.plot(thist[:], (np.sqrt(xhist[:, 2]**2 + xhist[:, 3]**2)))
plt.show()
