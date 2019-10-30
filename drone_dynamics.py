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
p1 = .12

x0 = np.array([1., 1., 0., 0., 0., 0.]) #posx, posy, angle, velox, veloy, angVel
u0 = (m*g / 2.) * np.ones((int(tfinal/dt), 2)) #thrust 1 and thrust 2
R = tf.linalg.diag([1., 1.])
Q = tf.linalg.diag([1., 1., 1., 1., 1., 1.])
Qf = 10*tf.linalg.diag([1., 1., 1., 1., 1., 1.])
xg = tf.constant([2., 2., 0., 0., 0., 0.]) #goal position

def deriv(xs, us):
    Ft = us[0] + us[1]
    vxdot = Ft * tf.sin(xs[2]) / m
    vydot = Ft * tf.cos(xs[2]) / m - g
    wdot = 2*(us[0]-us[1])/(m * l)
    return xs[3], xs[4], xs[5], vxdot, vydot, wdot

def graph_deriv(xs, us):
    Ft = us[0] + us[1]
    vxdot = Ft * math.sin(xs[2]) / m
    vydot = Ft * math.cos(xs[2]) / m - g
    wdot = 2*(us[0]-us[1])/(m * l)
    return xs[3], xs[4], xs[5], vxdot, vydot, wdot

def hist(x0, us, dt, tfinal): #for tensor
    t = 0.
    i = 0
    xhist = tf.constant(x0, shape=[1, 6, C1], dtype = tf.float32)
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
    t = 0
    i = 0
    thist = np.array([t]) #records time in respect to dt
    xhist = np.array([x0]) #records all previous positions
    t += dt
    while t < tfinal:
        x = xhist[-1]
        x = np.add(x, np.multiply(graph_deriv(x, us[i]), dt))
        i += 1
        t += dt
        thist = np.append(thist, np.array([t]), axis=-1)
        xhist = np.append(xhist, np.array([x]), axis=0)
        #print("xhist shape = ", xhist.shape)
        #print("thist shape = ", thist.shape)
    return xhist, thist

def quadratic_cost_for(x0, xg, us, Q, Qf, R, dt, tfinal): 
    xg = tf.expand_dims(xg, 0)
    u = tf.reshape(us, [int(int(us.shape[0])/2), 2, C1]) #reintroduce the thrusts from vectorized
    xn, tn = hist(x0, u, dt, tfinal)
    '''
    print('xg = ', xg)
    print('u = ', u)
    print('xn = ', xn)
    print('tn = ', tn)'''
    u = tf.transpose(u, perm = [2, 0, 1]) #matmul does batch mult with batch size infront
    xn = tf.transpose(xn, perm = [2, 0, 1])

    Q, Qf, R = map(lambda x: tf.tile(tf.expand_dims(x, 0), [C1, 1, 1]), [Q, Qf, R])

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
        print(np.mean(J_out))
        u_out = np.reshape(u_out, [int(np.size(u0)/2), 2, C])
        
        cost = np.sort(np.concatenate((u_out, J_out), axis = 1), axis = -1) #test this, might be wrong
        #print("sample with cost = ", cost.shape)
        cost = np.delete(cost, -1, 1)
        #print("sample w/o cost = ", cost.shape)
        
        e_samples = cost[:, :, :int(C*p)] #test this, might be wrong
        #print("elite = ", e_samples.shape)
        mu = np.reshape(np.mean(e_samples, axis = 2), [-1])
        e_samples = np.reshape(e_samples, (np.size(e_samples, axis = 0) * np.size(e_samples, axis = 1), np.size(e_samples, axis = 2)))
        #print(e_samples.shape)
        sigma = np.cov(e_samples)
        #print(sigma.shape)
    return mu, sigma

ur0 = np.reshape(u0, [-1])
u = tf.placeholder(tf.float32, shape = (ur0.shape[0], None))
sigma0 = 10*np.diag(np.ones_like(ur0))

cost = quadratic_cost_for(x0, xg, u, Q, Qf, R, dt, tfinal)
mu, sigma = CEM(ur0, cost, sigma0, L1, C1, p1)

'''xtest = tf.constant([1., 1., 0., 0., 0., 0.], dtype = tf.float64)
print(np.mean(quadratic_cost_for(x0, xtest, ur0, Q, Qf, R, dt, tfinal)))
'''
us = np.reshape(mu, [int(mu.size/2), 2])
#print("us = ", us.shape)
xhist, thist = graph_hist(x0, us, dt, tfinal)
#print(xhist)

plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.plot(xhist[:, 0], xhist[:, 1]) #position graph

plt.figure()
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity over Time')
plt.plot(thist[:], (np.sqrt(xhist[:, 3]**2 + xhist[:, 4]**2))) #velocity graph
plt.show()

'''
plt.figure()
plt.xlabel('Time')
plt.ylabel('Thrust')
plt.title('Thrust over Time')
plt.plot(thist[:], us[:, 0]) #thrust 1
plt.plot(thist[:], us[:, 1]) #thrust 2
plt.show()
'''
