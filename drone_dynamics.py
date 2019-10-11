import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

m = 10. #mass of drone
l = 10. #length of drone
g = 9.81
dt = 0.05
tfinal = 10

x0 = tf.constant([1., 1., 1., 1., 1., 1.]) #posx, posy, angle, velox, veloy, angVel
u0 = tf.zeros((int(tfinal/dt), 2), tf.float32) #thrust 1 and thrust 2
u = tf.placeholder(tf.float32, shape = (int(tfinal/dt)))
R = tf.linalg.diag([1., 1., 1.])
Q = tf.linalg.diag([1., 1., 1., 1., 1., 1.])
Qf = 10*tf.linalg.diag([1., 1., 1., 1., 1., 1.])
xg = tf.constant([2., 2., 2., 2., 2., 2.]) #goal position
L1 = 10
C1 = 100
p1 = .016

def deriv(xs, us):
    Ft = us[0] + us[1]
    vxdot = Ft * tf.sin(xs[2]) / m
    vydot = Ft * tf.cos(xs[2]) / m - g
    wdot = 2*(us[0]-us[1])/(m * l)
    return vxdot, vydot, wdot

def hist(x0, us, dt, tf):
    t = 0
    i = 0
    xhist = tf.constant([x0])
    thist = tf.constant([t])
    while t < tf:
        x += deriv(xhist[-1], us[i])*dt
        i += 1
        t += dt
        thist.append(t)
        xhist.append(x)
    return xhist, thist

def quadratic_cost_for(x0, xg, u, Q, Qf, R, dt, tf):
    u = tf.reshape(u, [int(u.size()/2), 2])
    xn, tn = hist(x0, u, dt, tf)
    cost = 0
    for i in range(0, tf.shape(u)[0]):
        cost += (tf.matmul(tf.matmul(u[i], R, transpose_a = True), u[i]) + tf.matmul(tf.matmul((xn[i]-xg), Q, transpose_a = True), (xn[i] - xg))) * dt
    cost += tf.matmul(tf.matmul((xn[-1]-xg), Qf, transpose_a = True), (xn[-1] - xg))
    return cost

def costSort(cost):
    return sample[1]

def CEM(u0, J, E0, L, C, p): #L is the maximum iterations done and C is samples per iteration
    mu = u0
    sigma = E0
    for l in range(0, L):
        state_cost = []
        u_out = np.random.multivariate_normal(mu, sigma, (u0.shape[0], C))
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

ur0 = tf.reshape(u0, [int(tfinal/dt), 1])
sigma0 = 10 * tf.linalg.diag(tf.ones_like(ur0))
cost = lambda us : quadratic_cost_for(us, R, Q, Qf, tf, dt, xg, x0)
sess = tf.Session()
mu, sigma = sess.run(CEM(u, cost, sigma0, L1, C1, p1), feed_dict = {u: ur0})

us = tf.reshape(mu, [int(mu.size/2), 2])
xhist, thist = hist(x0, us, dt, tf)

plt.figure()
plt.plot(xhist[:, 0], xhist[:, 1])
plt.figure()
plt.plot(thist[:], (tf.math.sqrt(xhist[:, 2]**2 + xhist[:, 3]**2)))
plt.figure()
plt.plot(thist[:-1], us[:, 0])
plt.plot(thist[:-1], us[:, 1])
plt.show()
