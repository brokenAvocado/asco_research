import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

m = 10. #mass of drone
l = 10. #length
g = 9.81
dt = 0.05
tfinal = 10.
L1 = 100
C1 = 100
p1 = .12

x0 = np.array([10., 5., 0., 0., 0., 0.]) #posx, posy, angle, velox, veloy, angVel
u0 = (m*g / 2.) * np.ones((int(tfinal/dt), 2)) #thrust 1 and thrust 2
R = .01 * tf.linalg.diag([1., 1.])
Q = 0 * tf.linalg.diag([1., 1., 1., 1., 1., 1.])
Qf = 10 * tf.linalg.diag([1., 1., 1., 1., 1., 1.])
xg = tf.constant([3., -9., 0., 0., 0., 0.]) #goal position

os0 = tf.constant([[4., 5.], [2., -1.]]) #obstacles
r = 4 #radius

def deriv(xs, us):
    Ft = us[0] + us[1]
    vxdot = Ft * tf.sin(xs[2]) / m
    vydot = (Ft * tf.cos(xs[2]) / m) - g
    wdot = 2*(us[0]-us[1])/(m * l)
    return xs[3], xs[4], xs[5], vxdot, vydot, wdot

def graph_deriv(xs, us):
    Ft = us[0] + us[1]
    vxdot = Ft * math.sin(xs[2]) / m
    vydot = (Ft * math.cos(xs[2]) / m) - g
    wdot = 2*(us[0]-us[1])/(m * l)
    return xs[3], xs[4], xs[5], vxdot, vydot, wdot

def hist(x0, us, dt, tfinal): #for tensor
    t = 0.
    i = 0
    xhist = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(x0, 0), 2), [1, 1, tf.shape(us)[2]]), tf.float32)
    thist = tf.constant([t], dtype = tf.float64)
    t += dt
    while t < tfinal:
        x = xhist[-1, :, :]
        x += tf.stack(deriv(x, us[i, :, :]), 0) * dt
        i += 1
        t += dt
        thist = tf.concat([thist, [t]], 0)
        x = tf.expand_dims(x, 0)
        xhist = tf.concat([xhist, x], 0)
    return xhist, thist

def graph_hist(x0, us, dt, tfinal): #for plotting
    t = 0.
    i = 0
    thist = np.array([t]) #records time in respect to dt
    xhist = np.array([x0]) #records all previous positions
    t += dt
    while t < tfinal:
        x = xhist[-1]
        x = np.add(x, np.multiply(graph_deriv(x, us[i]), dt))
        i += 1
        t += dt
        thist = np.append(thist, np.array([t]), axis=0)
        xhist = np.append(xhist, np.array([x]), axis=0)
    return xhist, thist

def quadratic_cost_for(x0, xg, us, Q, Qf, R, dt, tfinal, lam): 
    xg = tf.expand_dims(xg, 0)
    u = tf.reshape(us, [int(int(us.shape[0])/2), 2, tf.shape(us)[1]]) #reintroduce the thrusts from vectorized
    u = tf.cast(u, tf.float32)
    xn, tn = hist(x0, u, dt, tfinal)
    u = tf.transpose(u, perm = [2, 0, 1]) #matmul does batch mult with batch size infront
    xn = tf.transpose(xn, perm = [2, 0, 1])

    Q, Qf, R = map(lambda x: tf.tile(tf.expand_dims(x, 0), [C1, 1, 1]), [Q, Qf, R])

    state_mul = tf.matrix_diag_part(tf.matmul(tf.matmul(u, R), tf.transpose(u, perm = [0, 2, 1])))
    
    pos_mul = tf.matrix_diag_part(tf.matmul(tf.matmul((xn - xg), Q), tf.transpose((xn - xg), perm = [0, 2, 1])))
    mul_sum = tf.reduce_sum(state_mul + pos_mul, 1, True) * dt

    xf = tf.expand_dims(xn[:, -1, :], 1)
    mul_sum = tf.expand_dims(mul_sum, 1)
    termf = tf.matmul(tf.matmul((xf - xg), Qf), tf.transpose((xf - xg), perm = [0, 2, 1]))
    
    cost = mul_sum + termf
    cost = tf.transpose(cost, perm = [1, 2, 0])
    cost += lam*constraint(xn, os0, r)
    return cost

def constraint(xs, os, r): #finds the penalty based on the radius minus distance from object pos to center of obstacle
    penalty = 0
    for n in range(0, int(os.shape[0])):
        cval = r - tf.norm(xs[:, :2, :] - tf.expand_dims(tf.expand_dims(os[n, :], 0), 2), axis = 1, keep_dims = True)
        cval = tf.cast(cval >= 0, cval.dtype) * cval
        penalty += cval
    return tf.reduce_sum(penalty, 0, True)

def costSort(cost):
    return sample[1]

def CEM(sess, u0, J, E0, L, C, p): #L is the maximum iterations done and C is samples per iteration
    mu = u0
    sigma = E0
    lam0 = 2
    for l in range(0, L):
        lam0 = lam0 ** l
        u_out = np.random.multivariate_normal(mu, sigma, size = C).T
        J_out = sess.run(J, feed_dict = {u: u_out, lam: lam0})
        print(np.mean(J_out))
        u_out = np.reshape(u_out, [int(np.size(u0)/2), 2, C])
        J_out = np.repeat(J_out, np.size(u_out, 0), 0)
        
        cost = np.array(np.append(u_out, J_out, 1))
        cost = cost[:, :, (cost[:, -1, :].argsort()[0])]
        cost = np.delete(cost, -1, 1)

        e_samples = cost[:, :, :int(C*p)] #test this, might be wrong
        #print("elite = ", e_samples.shape)
        mu = np.reshape(np.mean(e_samples, axis = 2), [-1])
        e_samples = np.reshape(e_samples, (np.size(e_samples, axis = 0) * np.size(e_samples, axis = 1), np.size(e_samples, axis = 2)))
        #print(e_samples.shape)
        sigma = np.cov(e_samples) + 0.01 * np.diag(np.ones_like(mu)) 
        #print(sigma.shape)
    return mu, sigma
    
sess = tf.Session()
ur0 = np.reshape(u0, [-1])
u = tf.placeholder(tf.float32, shape = (ur0.shape[0], None)) #change to none
xs = tf.placeholder(tf.float32, shape = (ur0.shape[0]/2, 2, C1))
lam = tf.placeholder(tf.float32)
sigma0 = 10*np.diag(np.ones_like(ur0))

cost = quadratic_cost_for(x0, xg, u, Q, Qf, R, dt, tfinal, lam)
mu, sigma = CEM(sess, ur0, cost, sigma0, L1, C1, p1)

us = np.reshape(mu, [int(mu.size/2), 2]) #change back to mu
_, thist = graph_hist(x0, us, dt, tfinal)
tf_xhist, tf_thist = hist(x0, us[:, :, np.newaxis], dt, tfinal)
xhist = sess.run(tf_xhist)

plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.plot(xhist[:, 0], xhist[:, 1]) #position graph

plt.figure()
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity over Time')
#plt.plot(thist[:], (np.sqrt(xhist[:, 3]**2 + xhist[:, 4]**2))) #velocity graph
plt.plot(thist[:], xhist[:, 4])
plt.show()

plt.figure()
plt.xlabel('Time')
plt.ylabel('Thrust')
plt.title('Thrust over Time')
plt.plot(thist[:], us[:, 0]) #thrust 1
plt.plot(thist[:], us[:, 1]) #thrust 2
plt.show()

