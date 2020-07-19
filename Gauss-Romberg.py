import tensorflow as tf
import tensorflow_probability as tfp
import time
import numpy as np
import csv
import psutil as ps
from math import pi

#Legendre-Gauss weights and nodes
with open('D:/University files/Third year/Second semester/Thesis/Legendre Gauss Nodes.csv',newline='') as Nodes:
    Nodes_data = csv.reader(Nodes,delimiter=',')
    Nodes_list = list(Nodes_data)
    Nodes_array = []
    for i in range(len(Nodes_list)):
        a=Nodes_list[i]
        b=[float(j) for j in a]
        Nodes_array.append(b)
    Legendre_Gauss_nodes = tf.convert_to_tensor(Nodes_array,dtype=tf.float64)
with open('D:/University files/Third year/Second semester/Thesis/Legendre Gauss Weights.csv',newline='') as weights:
   weights_data = csv.reader(weights,delimiter=',')
   weights_list = list(weights_data)
   weights_array = []
   for i in range(len(weights_list)):
       a = weights_list[i]
       b = [float(j) for j in a]
       weights_array.append(b)
   Legendre_Gauss_weights = tf.convert_to_tensor(weights_array, dtype=tf.float64)


@tf.function(autograph=False)
def integrand(x):
    return 1/(1 + 2500* x**2)

#evaluation function
@tf.function(autograph=False)
def Legendre_Gauss_eval(l,u,n):
    global Legendre_Gauss_weights
    global Legendre_Gauss_nodes
    w = tf.cast(Legendre_Gauss_weights[n-1],dtype=tf.float64)
    x = tf.cast(Legendre_Gauss_nodes[n-1],dtype=tf.float64)
    integral = ((u-l)/2)*tf.reduce_sum(w*integrand(((u-l)/2)*x +(u+l)/2))
    return integral

#Construct rows of Romberg matrix
@tf.function(autograph=False)
def add_element(r,x):
    r = tf.concat([r,x],axis=1)
    return r

@tf.function(autograph=False)
def complete_row(r,j):
    r = tf.concat([r,tf.zeros((1,98-j),dtype=tf.float64)],axis=1)
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
    return tf.less(tf.shape(r)[1],99)

@tf.function(autograph=False)
def row_builder(a,r):
    ans = tf.while_loop(row_builder_cond,
                        row_builder_body,
                        loop_vars=[a,r],
                        shape_invariants=[tf.TensorShape([None,None]),tf.TensorShape([None,None])])
    return ans[1]

#Add rows to the matrix (Legendre_Gauss)
@tf.function(autograph=False)
def add_first_row(l,u,tol,a):
    a_k0 = tf.reshape(Legendre_Gauss_eval(l,u,tf.cast(2,dtype=tf.int32)),shape=(1,1))
    a = tf.concat([a_k0, tf.zeros((1, 98), dtype=tf.float64)],axis=1)
    return a

@tf.function(autograph=False)
def add_other_rows(l,u,tol,a):
    k = tf.shape(a)[0]
    a_k0 = tf.reshape(Legendre_Gauss_eval(l , u, tf.cast(2+k,dtype=tf.int32)), shape=(1, 1))
    row = row_builder(a,a_k0)
    a = tf.concat([a,row],axis=0)
    return a

#Romberg function
@tf.function(autograph=False)
def romberg_body(l,u,tol,a):
    a = tf.cond(tf.not_equal(tf.shape(a)[1],99), lambda: add_first_row(l,u,tol,a),lambda :add_other_rows(l,u,tol,a))
    return l,u,tol,a

@tf.function(autograph=False)
def romberg_cond(l,u,tol,a):
    k = tf.shape(a)[0]
    cond = tf.cond(tf.not_equal(tf.shape(a)[1],99),
                   lambda: tf.constant(True),
                   lambda: tf.cond(tf.math.less(k,99),lambda: tf.cond(tf.math.greater_equal(tol, 1e-8),
                                   lambda: tf.cond(tf.math.greater_equal(tol, 1e-6),
                                                   lambda: tf.math.less(tol, tf.abs(
                                                       tf.subtract(a[k - 1, k - 1], a[k - 2, k - 2])) * 0.04),
                                                   lambda: tf.math.less(tol, tf.abs(tf.subtract(a[k - 1, k - 1], a[
                                                       k - 2, k - 2])))),
                                   lambda: tf.math.less(tol,
                                                        tf.abs(tf.subtract(a[k - 1, k - 1], a[k - 2, k - 2]) ))
                                   ),lambda :tf.constant(False))
                   )

    return cond

@tf.function(autograph=False)
def romberg(l,u,tol):
    res = tf.while_loop(romberg_cond,
                        romberg_body,
                        loop_vars=[tf.cast(l,dtype=tf.float64),
                                   tf.cast(u,dtype=tf.float64),
                                   tf.cast(tol,dtype=tf.float64),
                                   tf.zeros((1,1) ,dtype=tf.float64)],
                        shape_invariants= [tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape(()),
                                           tf.TensorShape([None,None])])
    mat = res[3]
    k = tf.shape(mat)[0]
    rel_error = mat[k-1,k-1]-mat[k-1,k-2]
    return mat[k-1,k-1],rel_error,k