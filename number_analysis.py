import numpy as np
import numpy.linalg as lin
import tensorflow as tf

''' Only REAL NUMBER is considered as a solution,
    solutions in complex field is nonsense.
    For a real symmetric matrix, all the eigen 
    values are real, and all the eigen vectors
    are real. Every pair of the eigen vector
    are orthogonal.
'''

''' Solving eigen values of a matrix using
    numpy linear algorithm tools.
'''
def NP_Eigen(mat):
    assert mat.shape[0]==mat.shape[1]
    return lin.eig(mat)

''' Normalizing vector with L2 metrics by row. 
'''
def TF_L2_Norm(m):
    t_scale = tf.sqrt(tf.reduce_sum(m*m))
    t_sign = m[0]/tf.abs(m[0])
    t_scale = 1.0/tf.maximum(t_scale, 1e-7)
    tf.stop_gradient(t_scale)
    tf.stop_gradient(t_sign)
    return t_sign*t_scale*m

''' Solving eigen values of a matrix using
    stochastic gradient descent optimizer.
    mat: the matrix to solve
    N: the maximum number of iteration
    M: the maximum number of jump
    R: the effective radius of tabu spots
'''
def GD_Eigen(mat, N, M, R):
    assert mat.shape[0]==mat.shape[1]
    eigens = np.zeros([mat.shape[0]])
    vectors = np.zeros(mat.shape)
    t_A = tf.constant(mat)
    t_lambda = tf.Variable(tf.random_uniform([1])-0.5)
    t_x = tf.Variable(tf.random_uniform([mat.shape[1],1])-0.5)
    t_x = TF_L2_Norm(t_x)
    t_err_pure = tf.matmul(t_A, t_x) - (t_lambda*t_x)
    t_err_pure = tf.reduce_mean(t_err_pure*t_err_pure)
    t_reg = tf.constant(0, tf.float32)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  
    for counter in xrange(mat.shape[0]):
        if counter>0:
            t_dist = tf.reduce_mean(tf.square(tf.reshape(t_x,[t_x.shape[0]*t_x.shape[1]])-vectors[counter-1]))
            t_cond = tf.cast(t_dist<R*R, tf.float32)
            tf.stop_gradient(t_cond)
            t_reg_add = 1.0/tf.maximum(t_dist, 1e-7)
            t_reg_add = t_cond*t_reg_add
            t_reg = t_reg + t_reg_add
        t_err = t_err_pure + t_reg
        trainer = opt.minimize(t_err)
        err = 1e7
        err_last = 0
        i = 0
        j = 0
        while i<N and j<M:
            i += 1
            err_last = err
            _, err, reg = sess.run([trainer, t_err, t_reg])
            if err < 1e-12:
                print 'Root#', counter, ' found at iter#', i
                print 'Solution is: ', sess.run([t_lambda, t_x])
                break
            elif abs(err_last - err) < 1e-15:
                print 'Trapped! Reinitialize vars and search again!'
                sess.run(tf.global_variables_initializer())
                i = 0
                j += 1
            elif err/(err_last-err)>N-i:
                print 'Optimize too slow at speed: ', err_last-err
                sess.run(tf.global_variables_initializer())
                i = 0
                j += 1
            #print err, reg
        # saving this solution
        eigens[counter], x = sess.run([t_lambda, t_x])
        vectors[counter] = x.reshape([mat.shape[1]])
    return eigens, vectors.T

def main():
    tf.set_random_seed(128)
    A = np.array([[1,3,-4],[3,-2,5],[-7,6,0]], dtype=np.float32)
    A = A.dot(A.T)
    A = A/np.sqrt(np.sum(A*A))
    s, u = GD_Eigen(A, 10000, 100, 1e-1)
    print s
    print u
    s, u = NP_Eigen(A)
    print s
    print u

main()

