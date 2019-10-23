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
L1 = 10
C1 = 100
p1 = .012

x0 = np.array([1., 1., 1., 1., 1., 1.]) #posx, posy, angle, velox, veloy, angVel
u0 = np.zeros((int(tfinal/dt), 2)) #thrust 1 and thrust 2
R = tf.constant([1., 1.])
Q = tf.constant([1., 1., 1., 1., 1., 1.])
Qf = 10*tf.constant([1., 1., 1., 1., 1., 1.])
xg = tf.constant([2., 2., 2., 2., 2., 2.]) #goal position

def deriv(xs, us):
    Ft = us[0] + us[1]
    vxdot = Ft * tf.sin(xs[2]) / m
    vydot = Ft * tf.cos(xs[2]) / m - g
    wdot = 2*(us[0]-us[1])/(m * l)
    return tf.stack([xs[3], xs[4], xs[5], vxdot, vydot, wdot], 0)

def hist(x0, us, dt, tfinal):
    t = 0.
    i = 0
    xhist = tf.constant(x0, shape=[1, 6, C1], dtype = tf.float32)
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
    xn = tf.transpose(xn, perm = [2, 0, 1])
    R = tf.linalg.diag(tf.reshape(tf.tile(R, [100]), (100, int(R.shape[0]))))
    Q = tf.linalg.diag(tf.reshape(tf.tile(Q, [100]), (100, int(Q.shape[0]))))
    Qf = tf.linalg.diag(tf.reshape(tf.tile(Qf, [100]), (100, int(Qf.shape[0]))))
    cost = (tf.matmul(tf.matmul(u, R), tf.transpose(u, perm = [0, 2, 1])) + tf.matmul(tf.matmul((xn - xg), Q), tf.transpose((xn - xg), perm = [0, 2, 1]))) * dt
    cost += tf.matmul(tf.matmul((xn - xg), Qf), tf.transpose((xn - xg), perm = [0, 2, 1]))
    cost = tf.expand_dims(tf.reduce_sum(tf.transpose(cost, perm = [1, 2, 0]), 1), 1)
    return cost 

def costSort(cost):
    return sample[1]

def CEM(u0, J, E0, L, C, p): #L is the maximum iterations done and C is samples per iteration
    mu = u0
    sigma = E0
    sess = tf.Session()
    for l in range(0, L):
        u_out = np.random.multivariate_normal(mu, sigma, size = C).T
        J_out = sess.run(J, feed_dict = {u: u_out})
        u_out = np.reshape(u_out, [int(np.size(u0)/2), 2, C])
        cost = np.sort(np.concatenate((u_out, J_out), axis = 1), axis = -1) #test this, might be wrong
        cost = np.delete(cost, -1, 1)
        e_samples = np.delete(cost, slice(None, int(C*p)), axis = 2) #test this, might be wrong
        mu = np.reshape(np.mean(e_samples, axis = 2), [-1])
        e_samples = np.reshape(e_samples, (np.size(e_samples, axis = 0)*np.size(e_samples, axis = 2), np.size(e_samples, axis = 1)))
        print(e_samples.shape)
        sigma = np.cov(e_samples)
        print(sigma.shape)
    return mu, sigma

ur0 = np.reshape(u0, [-1])
u = tf.placeholder(tf.float32, shape = (ur0.shape[0], None))
sigma0 = 10*np.diag(np.ones_like(ur0))
print(sigma0.shape)
print(ur0.shape)
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
