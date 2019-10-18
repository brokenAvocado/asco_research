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

x0 = np.array([1., 1., 1., 1., 1., 1.]) #posx, posy, angle, velox, veloy, angVel
u0 = np.zeros((int(tfinal/dt), 2)) #thrust 1 and thrust 2
R = tf.linalg.diag([1., 1.])
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
    return tf.stack([xs[3], xs[4], xs[5], vxdot, vydot, wdot], 0)

def hist(x0, us, dt, tfinal):
    t = 0.
    i = 0
    xhist = tf.constant(x0, shape=[1, 6, 100], dtype = tf.float32)
    thist = tf.constant([t])
    t += dt
    while t < tfinal:
        x = xhist[-1, :, :]
        x += deriv(x, us[i, :, :]) * dt
        i += 1
        t += dt
        thist = tf.concat([thist, [t]], 0)
        x = tf.expand_dims(x, 0)
        xhist = tf.concat([xhist, x], 0) 
    return xhist, thist

def quadratic_cost_for(x0, xg, us, Q, Qf, R, dt, tfinal): 
    xg = tf.expand_dims(xg, 0)
    u = tf.reshape(us, [int(int(us.shape[0])/2), 2, C1])
    xn, tn = hist(x0, u, dt, tfinal)
    u = tf.transpose(u, perm = [2, 0, 1])
    u = tf.expand_dims(u, 2)
    print(u.shape)
    xn = tf.transpose(xn, perm = [2, 0, 1])
    xn = tf.expand_dims(xn, 1)
    cost = 0
    for i in range(0, u.get_shape()[1]):
        cost += (tf.matmul(tf.matmul(u[:, i, :, :], R), u[:, i, :]) + tf.matmul(tf.matmul((xn[:, :, i, :] - xg[0, :]), Q), (xn[:, :, i, :] - xg[0, :]))) * dt
        print(cost.shape)
    cost += tf.matmul(tf.matmul((xn[:, :, -1, :]-xg[0, :]), Qf), (xn[:, :, -1, :] - xg[0, :]))
    print(cost.shape)
    return cost 

def costSort(cost):
    return sample[1]

def CEM(u0, J, E0, L, C, p): #L is the maximum iterations done and C is samples per iteration
    mu = u0
    sigma = E0
    sess = tf.Session()
    for l in range(0, L):
        u_out = np.random.multivariate_normal(mu, sigma, size = C)
        J_out = sess.run(J, feed_dict = {u: u_out.T})
        print(J_out.shape)
        e_samples = np.array([sc[0] for sc in state_cost[:int(p*len(state_cost))]])
        costs = np.array([sc[1] for sc in state_cost])
        print(np.mean(costs))
        mu = np.mean(e_samples.T, axis = 1)
        sigma = np.cov(e_samples.T)
    return mu, sigma

ur0 = np.reshape(u0, [-1])
u = tf.placeholder(tf.float32, shape = (ur0.shape[0], None))
sigma0 = 10*np.diag(np.ones_like(ur0))
cost = quadratic_cost_for(x0, xg, u, Q, Qf, R, dt, tfinal)
mu, sigma = CEM(ur0, cost, sigma0, L1, C1, p1)

us = np.reshape(mu, [int(mu.size/2), 2])
xhist, thist = hist(x0, us, dt, tf)

plt.figure()
plt.plot(xhist[:, 0], xhist[:, 1])
plt.figure()
plt.plot(thist[:], (tf.math.sqrt(xhist[:, 2]**2 + xhist[:, 3]**2)))
plt.figure()
plt.plot(thist[:-1], us[:, 0])
plt.plot(thist[:-1], us[:, 1])
plt.show()
