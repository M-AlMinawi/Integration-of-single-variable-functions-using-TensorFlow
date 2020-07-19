import tensorflow as tf
import time
import tensorflow_probability as tfp
import numpy as np
import csv
import psutil as ps
from math import pi

integrand = lambda x: 1/(1 + 250* x**2)
@tf.function(autograph=False)
def trapezium(l,u,n):
    #adjust x dtype according to start dtype
    x = tf.linspace(start=l, stop=u, num=tf.cast(n,dtype=tf.int32))
    dx = (x[-1] - x[0]) / (tf.cast(n,dtype=tf.float64) - 1)
    y = integrand(x)
    integral = ((y[0] + y[-1]) / 2 + tf.reduce_sum(y[1:-1])) * dx
    return integral

#Construct rows of Romberg matrix
@tf.function(autograph=False)
def add_element(r,x):
    r = tf.concat([r,x],axis=1)
    return r

@tf.function(autograph=False)
def complete_row(r,j):
    r = tf.concat([r,tf.zeros((1,24-j),dtype=tf.float64)],axis=1)
    return r

@tf.function(autograph=False)
def row_builder_body(a,r):
    k = tf.shape(a)[0]
    j = tf.shape(r)[1] - 1
    a_kj = tf.reshape(((4**(tf.cast(j,dtype=tf.float64)+1)*r[0,j]-a[k-1, j])/(4**(tf.cast(j,dtype=tf.float64)+1) - 1)), shape=(1,1))
    r = tf.cond(tf.less(j,k),
            true_fn=lambda : add_element(r,a_kj),
            false_fn=lambda : complete_row(r,j))
    return a,r

@tf.function(autograph=False)
def row_builder_cond(a,r):
    return tf.less(tf.shape(r)[1],25)

@tf.function(autograph=False)
def row_builder(a,r):
    ans = tf.while_loop(row_builder_cond,
                        row_builder_body,
                        loop_vars=[a,r],
                        shape_invariants=[tf.TensorShape([None,None]),tf.TensorShape([None,None])])
    return ans[1]

#Add rows to the matrix (Trapezium)
@tf.function(autograph=False)
def add_first_row(l,u,tol,a):
    a_k0 = tf.reshape(trapezium(l,u,tf.cast(2,dtype=tf.int32)),shape=(1,1))
    a = tf.concat([a_k0, tf.zeros((1, 24), dtype=tf.float64)],axis=1)
    return a

@tf.function(autograph=False)
def add_other_rows(l,u,tol,a):
    k = tf.shape(a)[0]
    a_k0 = tf.reshape(trapezium(l , u, tf.cast(2**(k + 1),dtype=tf.int32)), shape=(1, 1))
    row = row_builder(a,a_k0)
    a = tf.concat([a,row],axis=0)
    return a

#Romberg function
@tf.function(autograph=False)
def romberg_body(l,u,tol,a,min_iter):
    a = tf.cond(tf.not_equal(tf.shape(a)[1],25), lambda: add_first_row(l,u,tol,a),lambda :add_other_rows(l,u,tol,a))
    return l,u,tol,a,min_iter

@tf.function(autograph=False)
def romberg_cond(l,u,tol,a,min_iter):
    k = tf.shape(a)[0]
    cond = tf.cond(tf.less(tf.shape(a)[0],min_iter),
                   lambda: tf.constant(True),
                   lambda:tf.cond(tf.math.greater_equal(tf.shape(a)[0],25),
                                  lambda: tf.constant(False),
                                  lambda:tf.cond(tf.math.greater_equal(tol,1e-8),
                                                 lambda:tf.cond(tf.math.greater_equal(tol,1e-6),
                                                                lambda:tf.math.less(tol,tf.abs(tf.subtract(a[k-1,k-1],a[k-2,k-2])/200)),
                                                                lambda: tf.math.less(tol,tf.abs(tf.subtract(a[k - 1, k - 1], a[
                                                                                            k - 2, k - 2]) ))),
                                                 lambda: tf.math.less(tol,tf.abs(tf.subtract(a[k-1,k-1],a[k-2,k-2])/20))
                                  )))

    return cond

@tf.function(autograph=False)
def romberg(l,u,tol,min_iter=0):
    res = tf.while_loop(romberg_cond,
                        romberg_body,
                        loop_vars=[tf.cast(l,dtype=tf.float64),
                                   tf.cast(u,dtype=tf.float64),
                                   tf.cast(tol,dtype=tf.float64),
                                   tf.zeros((1,1) ,dtype=tf.float64)
                                   ,tf.cast(min_iter,dtype=tf.int32)],
                        shape_invariants= [tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape([None,None]),
                                           tf.TensorShape(())])
    mat = res[3]
    k = tf.shape(mat)[0]
    rel_error = mat[k-1,k-1]-mat[k-1,k-2]

    return mat[k-1,k-1],rel_error,k
