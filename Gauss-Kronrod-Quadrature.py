import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import csv
import psutil as ps
from math import pi

with open('D:/University files/Third year/Second semester/Thesis/Kronrod Gauss Nodes.csv',newline='') as Nodes:
   Nodes_data = csv.reader(Nodes,delimiter=',')
   Nodes_list = list(Nodes_data)
   Nodes_array = []
   for i in range(len(Nodes_list)):
       a=Nodes_list[i]
       b=[float(j) for j in a]
       Nodes_array.append(b)
   Kronrod_Gauss_nodes = tf.convert_to_tensor(Nodes_array,dtype=tf.float64)
with open('D:/University files/Third year/Second semester/Thesis/Kronrod Gauss Weights.csv',newline='') as weights:
  weights_data = csv.reader(weights,delimiter=',')
  weights_list = list(weights_data)
  weights_array = []
  for i in range(len(weights_list)):
      a = weights_list[i]
      b = [float(j) for j in a]
      weights_array.append(b)
  Kronrod_Gauss_weights = tf.convert_to_tensor(weights_array, dtype=tf.float64)

integrand = lambda x: x**2

def Kronrod_Gauss_eval(l,u,n):
    w = tf.cast(Kronrod_Gauss_weights[n],dtype=tf.float64)
    x = tf.cast(Kronrod_Gauss_nodes[n],dtype=tf.float64)

    integral = ((u-l)/2)*tf.reduce_sum(w*integrand(((u-l)/2)*x +(u+l)/2))

    return integral

def Kronrod_Gauss_body(l,u,n,tol,prev,current,max_points):
    prev = current
    current = Kronrod_Gauss_eval(l,u,n)
    n += 1
    return l,u,n,tol,prev,current,max_points

def Kronrod_Gauss_cond(l,u,n,tol,prev,current,max_points):
    cond = tf.cond(tf.greater_equal(n,max_points)
                   ,true_fn= lambda: tf.constant(False)
                   , false_fn=lambda: tf.cond(tf.math.greater_equal(tol, 1e-8),
                                              lambda: tf.cond(tf.math.greater_equal(tol, 1e-6),
                                                              lambda: tf.math.less(tol, tf.abs(
                                                                  tf.subtract(current, prev))),
                                                              lambda: tf.math.less(tol, tf.abs(
                                                                  tf.subtract(current, prev)))),
                                              lambda: tf.math.less(tol, tf.abs(tf.subtract(current, prev)))))
    return cond

def Kronrod_Gauss(l,u,tol,max_points=Kronrod_Gauss_weights.shape[0]):
    ans = tf.while_loop(Kronrod_Gauss_cond,Kronrod_Gauss_body,
                        loop_vars=[tf.cast(l,dtype=tf.float64),
                                   tf.cast(u,dtype=tf.float64),
                                   tf.cast(1,dtype=tf.int32),
                                   tf.cast(tol,dtype=tf.float64),
                                   tf.cast(0,dtype=tf.float64),
                                   tf.cast(1,dtype=tf.float64),
                                   tf.cast(max_points,dtype=tf.int32)],
                        shape_invariants=[tf.TensorShape(()),
                                          tf.TensorShape(()),
                                          tf.TensorShape(()),
                                          tf.TensorShape(()),
                                          tf.TensorShape(()),
                                          tf.TensorShape(()),
                                          tf.TensorShape(())])
    return ans[5], ans[5]-ans[4],ans[2]

