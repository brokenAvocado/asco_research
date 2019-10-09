import tensorflow as tf
x0 = tf.constant([1., 1., 1., 1., 1., 1.]) #posx, posy, angle, velox, veloy, angVel
u0 = tf.constant([1., 1.]) #thrust 1 and thrust 2
x = tf.placeholder(tf.float32)
u = tf.placeholder(tf.float32)
m = 10. #mass of drone
l = 10. #length of drone
g = 9.81
dt = 0.05
tf = 10

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
    u = tf.reshape(u, (u.size/2, 2))
    xn, tn = hist(x0, u, dt, tf)
    cost = 0
    for i in range(0, tf.shape(u)[0]):
        cost += (tf.matmul(tf.matmul(u[i], R, transpose_a = True), u[i]) + tf.matmul(tf.matmul((xn[i]-xg), Q, transpose_a = True), (xn[i] - xg))) * dt
    cost += tf.matmul(tf.matmul((xn[-1]-xg), Qf, transpose_a = True), (xn[-1] - xg))
    return cost

def CEM(u0, J, E0, L, C, p):
    mu = u0
    E = E0
    for l in range(0, L):
        state_cost = []
        for c in range(0, C):
            u_out = np.random.multivariate_normal(mu, sigma)
