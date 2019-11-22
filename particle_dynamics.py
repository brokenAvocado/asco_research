import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

t = 0.
dt = .02 #MUST BE EQUAL TO PARTICLE_COST dt
tf = 5.0 #^^
us = np.zeros((int(tf/dt), 2))
R = np.diag([1., 1.])
Q = 10*np.diag([1., 1., 1., 1.])
Qf = 10*np.diag([1., 1., 1., 1.]) 
xg = np.array([5., 5., 0., 0.]) #goal position
x0 = np.array([2., 2.,0.,0.]) #initial position

L1 = 10 #number of iterations to run
C1 = 100 #number of samples per iterations
p1 = 0.15 #percent of samples to keep

def particle_dynamics(x, u):
    p1 = x[0] #current x pos
    p2 = x[1] #current y pos
    v1 = x[2] #current x velo
    v2 = x[3] #current y velo
    a1 = u[0] #current x accel
    a2 = u[1] #current y accel
    return np.array([v1, v2, a1, a2]) #xdot or resultant

def hist(x0, us, dt, tf):
    t = 0
    i = 0
    thist = np.array([t]) #records time in respect to dt
    xhist = np.array([x0]) #records all previous positions
    t += dt
    while t < tf:
        x = xhist[-1]
        x += particle_dynamics(x, us[i])*dt
        i += 1
        t += dt
        thist = np.append(thist, np.array([t]), axis=0)
        xhist = np.append(xhist, np.array([x]), axis=0)
    return xhist, thist

def quadratic_cost_for(x, u, Q, R, Qf, dt, xg):
    a = 0
    for i in range(0, np.size(u, 0)):
       a += (np.dot(np.dot(np.transpose(u[i]),R), u[i]) + np.dot(np.dot(np.transpose(x[i]-xg), Q), (x[i]-xg)))*dt
    a += np.dot(np.dot(np.transpose(x[-1]-xg), Qf), (x[-1]-xg))
    return a

def particle_cost(us, R, Q, Qf, Tf, dt, xg, x0):
    us = np.reshape(us, (int(us.size/2), 2))
    xn, tn = hist(x0, us, dt, tf)
    return quadratic_cost_for(xn, us, Q, R, Qf, dt, xg)

def costSort(sample):
    return sample[1]

def CEM(u0, J, E0, L, C, p): #L is the maximum iterations done and C is samples per iteration
    mu = u0
    sigma = E0
    for l in range(0, L):
        state_cost = []
        for c in range(0, C):
            u_out = np.random.multivariate_normal(mu, sigma)
            J_out = J(u_out)
            sample = (u_out, J_out)
            state_cost.append(sample)
        state_cost.sort(key = costSort)
        e_samples = np.array([sc[0] for sc in state_cost[:int(p*len(state_cost))]])
        costs = np.array([sc[1] for sc in state_cost])
        print(np.mean(costs))
        mu = np.mean(e_samples.T, axis = 1)
        sigma = np.cov(e_samples.T)
    return mu, sigma
                
us0 = np.concatenate(us,-1)
print(us0.shape)
sigma0 = 10*np.diag(np.ones_like(us0))

cost = lambda us : particle_cost(us, R, Q, Qf, tf, dt, xg, x0)
mu, sigma = CEM(us0, cost, sigma0, L1, C1, p1)

us = np.reshape(mu, (int(mu.size/2), 2))
xhist, thist = hist(x0, us, dt, tf)

plt.figure()
plt.plot(xhist[:, 0], xhist[:, 1])
plt.figure()
plt.plot(thist[:], (np.sqrt(xhist[:, 2]**2 + xhist[:, 3]**2)))
plt.figure()
plt.plot(thist[:-1], us[:, 0])
plt.plot(thist[:-1], us[:, 1])
plt.show()
