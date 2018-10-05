import numpy as np
import tensorflow as tf

''' 
Optimizer: search the best solution with the help
of a trap detector and objective modifier.
'''

INF=-1

class Optimizer(object):
    def __init__(
            self, 
            base_optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-3),
            error_level=1e-5,
            converge_level=1e-7,
            trap_level=10,
            gravity_radius=1e-3,
            merge_level=0):
        optimizer_types = [
                tf.train.GradientDescentOptimizer,
                tf.train.AdamOptimizer]
        is_legal_type = False
        for opt_type in optimizer_types:
            is_legal_type = is_legal_type or (opt_type==type(base_optimizer))
        if not is_legal_type:
            print('Base optimizer is not supported, use SGD or Adam!')
        self.base_optimizer = base_optimizer
        self.error_level = error_level
        self.converge_level = converge_level
        # how many times to wait before the trap detector being triggered
        self.trap_level = trap_level
        self.merge_level = merge_level
        self.wait_time = 0
        self.regs = []
        self.vars = tf.trainable_variables()
        self.gravity_radius = gravity_radius
    def minimize(self, objective):
        self.objective = objective
        self.current_error = INF
        self.last_error = INF
        self.minimizer = self.base_optimizer.minimize(objective)
        return self.minimizer
    def update(self, sess, err): # The err is the evaluation of the base objective without any regs
        if self.current_error == INF:
            self.current_error = err
        elif self.last_error == INF:
            self.last_error = self.current_error
            self.current_error = err
        else:
            delta_err = np.abs(err-self.current_error)
            self.current_error = err
            self.last_error = self.current_error
            if delta_err<self.converge_level:
                self.wait_time += 1
            else:
                self.wait_time = 0
            if self.wait_time >= self.trap_level:
                if self.current_error<self.error_level:
                    return None # satisifed solution found!
                else:
                    print('Trap detected!')
                    # add trap escaping regularization to the objective
                    reg = 0
                    with sess.as_default(): 
                        for v in self.vars:
                            dist = tf.reduce_mean(tf.square(v-v.eval()))
                            cond = tf.cast(dist<self.gravity_radius, tf.float32)
                            gravity = 1.0/tf.maximum(dist, 1e-7)
                            gravity = cond * gravity
                            reg = reg + gravity
                    self.regs.append(reg)
                    # now update the final objective using weighted regularizations
                    print('Objective updated!')
                    reg_overall = 0
                    for reg in self.regs:
                        reg_overall = reg_overall + reg
                    self.minimizer = self.base_optimizer.minimize(0.5*self.objective+0.5*reg_overall/len(self.regs))
                    self.wait_time = 0 
        return self.minimizer


