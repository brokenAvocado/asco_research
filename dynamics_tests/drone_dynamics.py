import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
import time

m = 10. #mass of drone
l = 10. #length
g = 9.81
dt = 0.05
tfinal = 10.
L1 = 10
C1 = 1000
p1 = .12

x0 = np.array([10., 5., 0., 0., 0., 0.]) #posx, posy, angle, velox, veloy, angVel
u0 = (m*g / 2.) * np.ones((int(tfinal/dt), 2)) #thrust 1 and thrust 2
R = .01 * tf.linalg.diag([1., 1.])
Q = .1 * tf.linalg.diag([1., 1., 1., 1., 1., 1.])
Qf = 10 * tf.linalg.diag([1., 1., 1., 1., 1., 1.])
xg = tf.constant([3., -9., 0., 0., 0., 0.]) #goal position

os0 = tf.constant([[6., -4.], [10., -1.], [3., -7.], [9., 3.]]) #obstacles
r = 1 #radius

tstart = time.perf_counter()

def deriv(xs, us, w):
    Ft = us[0] + us[1]
    vxdot = Ft * tf.sin(xs[2]) / m
    vydot = (Ft * tf.cos(xs[2]) / m) - g
    wdot = 2*(us[0]-us[1])/(m * l)

    noise_shape = tf.shape(us)[1], int(xs.shape[0]), int(xs.shape[0])
    noiseFx_shape = tf.shape(us)[1], int(int(xs.shape[0])/2), int(int(xs.shape[0])/2)

    #noiseFx = tf.sqrt(xs[3]**2 + xs[4]**2)*.0001 #+ (us[0] + us[1])*.0001
    zero = tf.zeros((tf.shape(us)[1], ))
    noiseFx = tf.transpose([[0.1 * tf.cos(xs[2]), -tf.sin(xs[2]), zero], [0.1 * tf.sin(xs[2]), tf.cos(xs[2]), zero], [zero, zero, zero]], perm = [2, 0, 1])
    #tf.cast(tf.reshape(tf.tile(noiseFx,[int(int(xs.shape[0])/2)**2]), noiseFx_shape), tf.float32) 
    noise_zeros = tf.zeros(noiseFx_shape)
    noise = tf.concat([noise_zeros, noise_zeros, noise_zeros, noiseFx], 0)
    noise = tf.reshape(noise, noise_shape)
    return tf.cast([xs[3], xs[4], xs[5], vxdot, vydot, wdot], tf.float32) + tf.transpose(tf.matmul(tf.expand_dims(tf.cast(w, tf.float32), 1), noise), perm = [1, 2, 0])

'''def graph_deriv(xs, us, w):
    Ft = us[0] + us[1]
    vxdot = Ft * math.sin(xs[2]) / m
    vydot = (Ft * math.cos(xs[2]) / m) - g
    wdot = 2*(us[0]-us[1])/(m * l)
    noiseFx = np.sqrt(xs[3]**2 + xs[4]**2)*.0001 #+ (us[0] + us[1])*.0001
    noise = np.reshape(np.repeat(noiseFx ,int(xs.shape[0])**2), (int(xs.shape[0]), int(xs.shape[0])))
    return [xs[3], xs[4], xs[5], vxdot, vydot, wdot] + np.matmul(noise, w.T)
'''
def hist(x0, us, dt, tfinal, w): #for tensor
    t = 0.
    i = 0
    xhist = tf.tile(tf.expand_dims(tf.expand_dims(x0, 0), 2), [1, 1, tf.shape(us)[2]])
    thist = tf.constant([t], dtype = tf.float64)
    t += dt
    while t < tfinal:
        x = xhist[-1, :, :]
        x += deriv(x, us[i, :, :], w[:, i, :]) * dt
        i += 1
        t += dt
        thist = tf.concat([thist, [t]], 0)
        xhist = tf.concat([xhist, x], 0)
    return xhist, thist

'''def graph_hist(x0, us, dt, tfinal, w): #for plotting
    t = 0.
    i = 0
    thist = np.array([t]) #records time in respect to dt
    xhist = np.array([x0]) #records all previous positions
    t += dt
    while t < tfinal:
        x = xhist[-1]
        x = np.add(x, np.multiply(graph_deriv(x, us[i], w[1, i, :]), dt))
        i += 1
        t += dt
        thist = np.append(thist, np.array([t]), axis=0)
        xhist = np.append(xhist, np.array([x]), axis=0)
    return xhist, thist
'''
def quadratic_cost_for(x0, xg, us, Q, Qf, R, dt, tfinal, lam, w): 
    xg = tf.expand_dims(xg, 0)
    u = tf.reshape(us, [int(int(us.shape[0])/2), 2, tf.shape(us)[1]]) #reintroduce the thrusts from vectorized
    u = tf.cast(u, tf.float32)
    w = tf.convert_to_tensor(w, tf.float32)
    x0 = tf.cast(x0, tf.float32)
    xn, tn = hist(x0, u, dt, tfinal, w)
    u = tf.transpose(u, perm = [2, 0, 1]) #matmul does batch mult with batch size infront
    xn = tf.transpose(xn, perm = [2, 0, 1])

    Q, Qf, R = map(lambda x: tf.tile(tf.expand_dims(x, 0), [tf.shape(us)[1], 1, 1]), [Q, Qf, R])

    state_mul = tf.matrix_diag_part(tf.matmul(tf.matmul(u, R), tf.transpose(u, perm = [0, 2, 1])))
    
    pos_mul = tf.matrix_diag_part(tf.matmul(tf.matmul((xn - xg), Q), tf.transpose((xn - xg), perm = [0, 2, 1])))
    mul_sum = tf.reduce_sum(state_mul + pos_mul, 1, True) * dt

    xf = tf.expand_dims(xn[:, -1, :], 1)
    mul_sum = tf.expand_dims(mul_sum, 1)
    termf = tf.matmul(tf.matmul((xf - xg), Qf), tf.transpose((xf - xg), perm = [0, 2, 1]))
    
    cost = mul_sum + termf
    costraw = tf.transpose(cost, perm = [1, 2, 0])
    xn = tf.transpose(xn, perm = [1, 2, 0])
    constr = lam*constraint(xn, os0, r)
    return costraw + constr, costraw, constr/lam

def constraint(xs, os, r): #finds the penalty based on the radius minus distance from object pos to center of obstacle
    penalty = 0
    for n in range(0, int(os.shape[0])):
        #print(xs.shape)
        #print(xs[:, :2, :])
        cval = r - tf.norm(xs[:, :2, :] - tf.expand_dims(tf.expand_dims(os[n, :], 0), 2), axis = 1, keep_dims = True)
        #print(cval.shape)
        cval = tf.cast(cval >= 0, cval.dtype) * cval
        penalty += cval
    return tf.reduce_sum(penalty, 0, True)

def costSort(cost):
    return sample[1]

def CEM(sess, u0, J, E0, L, C, p): #L is the maximum iterations done and C is samples per iteration
    mu = u0
    sigma = E0
    for l in range(0, L):
        u_out = np.random.multivariate_normal(mu, sigma, size = C).T
        #print(u_out.shape)
        noise = np.random.multivariate_normal(np.zeros(int(x0.shape[0] * tfinal/dt)), np.identity(int(x0.shape[0] * tfinal/dt)), size = C) #noise
        noise = np.reshape(noise, (C, int(tfinal/dt), x0.shape[0]))
        J_out, cost_raw, constr = sess.run(J, feed_dict = {u: u_out, w: noise})
        print("J_out cost: ", np.mean(J_out))
        '''print("J_out raw cost: ", np.mean(cost_raw))
        print("J_out constraint w/o lambda: ", np.mean(constr))'''
        u_out = np.reshape(u_out, [int(np.size(u0)/2), 2, C])
        J_out = np.repeat(J_out, np.size(u_out, 0), 0)
        
        cost = np.array(np.append(u_out, J_out, 1))
        cost = cost[:, :, (cost[:, -1, :].argsort()[0])]
        cost = np.delete(cost, -1, 1)
        
        a_samples = cost
        e_samples = cost[:, :, :int(C*p)]
        #print("elite = ", e_samples.shape)
        mu = np.reshape(np.mean(e_samples, axis = 2), [-1])
        #J_out, costraw, constr = sess.run(J, feed_dict = {u: np.reshape(mu, [mu.size, 1])})
        '''print("mu cost: ", np.mean(J_out))
        print("mu raw cost: ", np.mean(cost_raw))
        print("mu constraint w/o lambda: ", np.mean(constr))'''
        e_samples = np.reshape(e_samples, (np.size(e_samples, axis = 0) * np.size(e_samples, axis = 1), np.size(e_samples, axis = 2)))
        #print(e_samples.shape)
        sigma = np.cov(e_samples) + 0.01 * np.diag(np.ones_like(mu)) 
        #print(sigma.shape)
    return mu, sigma, e_samples
    
sess = tf.Session()
ur0 = np.reshape(u0, [-1])
u = tf.placeholder(tf.float32, shape = (ur0.shape[0], None)) 
#xs = tf.placeholder(tf.float32, shape = (ur0.shape[0]/2, 2, C1))
w = tf.placeholder(tf.float32, shape = (None, int(tfinal/dt), x0.shape[0]))
sigma0 = 10*np.diag(np.ones_like(ur0))

lam0 = 10.0
lam = tf.Variable(lam0)
sess.run(lam.initializer)
cost = quadratic_cost_for(x0, xg, u, Q, Qf, R, dt, tfinal, lam, w)
for o in range(0, 10):
    sess.run(lam.assign(10.0**o))
    print("lambda: ", sess.run(lam))
    mu, sigma, samples = CEM(sess, ur0, cost, sigma0, L1, C1, p1)
    ur0 = np.reshape(mu, [-1])
    sigma0 = sigma
    print()

tstop = time.perf_counter()
print(tstop - tstart, "sec.")
us = np.reshape(samples, [int(mu.size/2), 2, int(samples.shape[1])]) #np.reshape(mu, [int(mu.size/2), 2]) for mean samples

usgraph = tf.cast(us, tf.float32) #tf.reshape(mu, [int(mu.size/2), 2, 1]), tf.float32)
x0 = tf.cast(x0, tf.float32)
noise = np.random.multivariate_normal(np.zeros(int(x0.shape[0]) * int(tfinal/dt)), np.identity(int(x0.shape[0]) * int(tfinal/dt)), size = int(C1*p1)) #noise
wfinal = np.reshape(noise, (int(C1*p1), int(tfinal/dt), x0.shape[0]))
xhist, thist = sess.run(hist(x0, usgraph, dt, tfinal, wfinal))
print("xhist: ", xhist.shape)
print("thist: ", thist.shape)

os0 = np.array(sess.run(os0)) #obstacles

plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.plot(xhist[:, 0, :], xhist[:, 1, :]) #position graph
ax = plt.gca()
for f in range(0, int(os0.shape[0])):
    ax.add_patch(plt.Circle(os0[f], r, color = 'r'))

plt.figure()
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity over Time')
#plt.plot(thist[:], (np.sqrt(xhist[:, 3]**2 + xhist[:, 4]**2))) #velocity graph
plt.plot(thist[:], xhist[:, 4, :])

plt.figure()
plt.xlabel('Time')
plt.ylabel('Thrust')
plt.title('Thrust over Time')
plt.plot(thist[:], us[:, 0, :]) #thrust 1
plt.plot(thist[:], us[:, 1, :]) #thrust 2
plt.show()
