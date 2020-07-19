import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from math import pi
import time
import matplotlib.pyplot as plt
import csv
import psutil as ps

#remove graphing edits before testing performance


def integrate(func, lower, upper):
    return (func(lower)  +  func(upper)) / 2 * (upper - lower), func(lower),func(upper)

initial_points = tf.linspace(tf.cast(0,dtype=tf.float64), tf.cast(3,dtype=tf.float64), num=800)
lower,upper = initial_points[:-1],initial_points[1:]


func = lambda x: 1/(1 + 250* x**2)



@tf.function(autograph=False)
def body(integral,increase,tol,lower,upper):
    integrals,low,up = integrate(func,lower,upper)
    gradients = tf.abs((up-low)/tf.abs(upper-lower))
    too_big = tf.greater(gradients,tf.reduce_mean(gradients))
    points  = tf.where(too_big)[:, 0]

    integral += tf.reduce_sum(tf.boolean_mask(integrals, mask=tf.logical_not(too_big)), axis=0)
    increase = tf.reduce_sum(tf.boolean_mask(integrals, mask=tf.math.equal(too_big,True)), axis=0)
    lower_to_redo = tf.gather(lower, points, axis=0)
    # tf.print(lower_to_redo[:5])
    upper_to_redo = tf.gather(upper, points, axis=0)
    # tf.print(upper_to_redo[:5])
    new_middle = (upper_to_redo + lower_to_redo) / 2
    new_lower = tf.concat([lower_to_redo, new_middle], axis=0)
    new_upper = tf.concat([new_middle, upper_to_redo], axis=0)
    return integral, increase,tol ,new_lower, new_upper

@tf.function(autograph=False)
def diff_body(integral,increase,tol,lower,upper,plot_points,rejected_points,iterations):
    integrals,low,up = integrate(func,lower,upper)
    abs_diff = tf.abs(up-low)
    abs_diff_bound = tf.cond(tf.math.greater(tol,tf.cast(1e-10,dtype=tf.float64)),lambda :tf.cond(tf.math.greater(tol,1e-6),
                                                                       lambda: tf.abs((tf.cast(1/4, dtype=tf.float64) * tol**(1/4))),
                                                                       lambda : tf.abs(8 * ((tol ** (3/ 4)))))
                             ,lambda :tf.abs(5000* ((tol))))
    too_big = tf.greater(abs_diff,abs_diff_bound)
    points  = tf.where(too_big)[:, 0]
    integral += tf.reduce_sum(tf.boolean_mask(integrals, mask=tf.logical_not(too_big)), axis=0)
    increase = tf.reduce_sum(tf.boolean_mask(integrals, mask=tf.math.equal(too_big,True)), axis=0)
    lower_to_redo = tf.gather(lower, points, axis=0)
    upper_to_redo = tf.gather(upper, points, axis=0)
    new_middle = (upper_to_redo + lower_to_redo) / 2
    new_lower = tf.concat([lower_to_redo, new_middle], axis=0)
    new_upper = tf.concat([new_middle, upper_to_redo], axis=0)
    to_plot = tf.concat([lower,tf.reshape(upper[-1],shape=(1,))],axis=0)
    accepted_points = tf.boolean_mask(to_plot,mask=tf.logical_not(too_big))
    rejected_points = tf.boolean_mask(to_plot,mask=too_big)
    plot_points = tf.concat([plot_points,accepted_points],axis=0)
    iterations += 1
    return integral, increase,tol ,new_lower, new_upper,plot_points,rejected_points,iterations

@tf.function(autograph=False)
def norm_body(integral,increase,tol,lower,upper):
    integrals,low,up = integrate(func,lower,upper)
    norm_diff = tf.sqrt(up**2 + upper**2) - tf.sqrt(low**2 + lower**2)
    too_big = tf.greater(norm_diff,tol*1e4/2)
    points  = tf.where(too_big)[:, 0]
    integral += tf.reduce_sum(tf.boolean_mask(integrals, mask=tf.logical_not(too_big)), axis=0)
    increase = tf.reduce_sum(tf.boolean_mask(integrals, mask=tf.math.equal(too_big,True)), axis=0)
    lower_to_redo = tf.gather(lower, points, axis=0)
    # tf.print(lower_to_redo[:5])
    upper_to_redo = tf.gather(upper, points, axis=0)
    # tf.print(upper_to_redo[:5])
    new_middle = (upper_to_redo + lower_to_redo) / 2
    new_lower = tf.concat([lower_to_redo, new_middle], axis=0)
    new_upper = tf.concat([new_middle, upper_to_redo], axis=0)
    return integral, increase ,tol ,new_lower, new_upper

@tf.function(autograph=False)
def cond(integral,increase,tol, lower, upper,plot_points,rejected_points,iterations):

    cond = tf.cond(tf.equal(integral,tf.cast(0,dtype=tf.float64))
                   ,lambda: tf.constant(True)
                   ,lambda :tf.greater(tf.abs(increase),tol))
    return cond


@tf.function(autograph=False)
def adaptive_trapezium(l,u,tol):
    initial_points = tf.linspace(tf.cast(l,dtype=tf.float64), tf.cast(u,dtype=tf.float64), num=100)
    result = tf.while_loop(cond=cond, body=body, loop_vars=[tf.cast(0.,dtype=tf.float64),
                                                                    tf.cast(0, dtype=tf.float64),
                                                                    tf.cast(tol,dtype=tf.float64),
                                                                    initial_points[:-1],
                                                                    initial_points[1:]
                                                                    ],
                         shape_invariants=[tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape((None,)),
                                           tf.TensorShape((None,)),

                                           ])
    integral = result[0] + result[1]
    return integral


@tf.function(autograph=False)
def adaptive_trapezium_diff(l,u,tol,iter):
    initial_points = tf.linspace(tf.cast(l,dtype=tf.float64), tf.cast(u,dtype=tf.float64), num=45)
    result = tf.while_loop(cond=cond, body=diff_body, loop_vars=[tf.cast(0.,dtype=tf.float64),
                                                                    tf.cast(0, dtype=tf.float64),
                                                                    tf.cast(tol,dtype=tf.float64),
                                                                    initial_points[:-1],
                                                                    initial_points[1:],
                                                                    tf.reshape(tf.cast(u,dtype=tf.float64),shape=(1,)),
                                                                 tf.zeros((1,), dtype=tf.float64),
                                                                 tf.cast(0,dtype=tf.float64)
                                                                    ],
                         shape_invariants=[tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape((None,)),
                                           tf.TensorShape((None,)),
                                           tf.TensorShape((None,)),
                                           tf.TensorShape((None,)),
                                           tf.TensorShape((None,)),

                                           ], maximum_iterations=iter)

    integral = result[0] + result[1]
    plotting = tf.concat([result[5],result[6][1:]],axis=0)
    iterations = result[7]

    return integral,plotting,iterations


@tf.function(autograph=False)
def adaptive_trapezium_norm(l,u,tol):
    initial_points = tf.linspace(tf.cast(l,dtype=tf.float64), tf.cast(u,dtype=tf.float64), num=100)
    result = tf.while_loop(cond=cond, body=norm_body, loop_vars=[tf.cast(0.,dtype=tf.float64),
                                                                    tf.cast(0, dtype=tf.float64),
                                                                    tf.cast(tol,dtype=tf.float64),
                                                                    initial_points[:-1],
                                                                    initial_points[1:]
                                                                    ],
                         shape_invariants=[tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape((None,)),
                                           tf.TensorShape((None,)),

                                           ])
    integral = result[0]+result[1]
    return integral