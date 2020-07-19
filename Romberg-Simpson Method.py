import tensorflow as tf
from math import pi
import time
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import csv
import psutil as ps

integrand = lambda x: 1/(1 + 250* x**2)

#Simpson's rule
@tf.function(autograph=False)
def simpsons(l,u,n):
    x = tf.linspace(start=l, stop=u, num=tf.cast(n, dtype=tf.int32))
    dx = (x[-1] - x[0]) / (tf.cast(n, dtype=tf.float64) - 1)
    y = integrand(x)
    integral = ((y[0]+y[-1])+2*tf.reduce_sum(y[1:-1:2])+4*tf.reduce_sum(y[2:-1:2]))*(dx/3)
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
    a_k0 = tf.reshape(simpsons(l,u,tf.cast(2,dtype=tf.int32)),shape=(1,1))
    a = tf.concat([a_k0, tf.zeros((1, 24), dtype=tf.float64)],axis=1)
    return a

@tf.function(autograph=False)
def add_other_rows(l,u,tol,a):
    k = tf.shape(a)[0]
    a_k0 = tf.reshape(simpsons(l , u, tf.cast(2**(k + 1),dtype=tf.int32)), shape=(1, 1))
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
                                                                lambda:tf.math.less(tol,tf.abs(tf.subtract(a[k-1,k-1],a[k-2,k-2]))*0.02),
                                                                lambda: tf.math.less(tol,tf.abs(tf.subtract(a[k - 1, k - 1], a[
                                                                                            k - 2, k - 2])*0.065 ))),
                                                 lambda: tf.math.less(tol,tf.abs(tf.subtract(a[k-1,k-1],a[k-2,k-2])*0.075))
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

    return mat[k-1,k-1],k

#Testing code

# muons = False
muons = True
if muons:
    SigShape = lambda loc, scale: tfd.Normal(loc, scale)
else:
    SigShape = lambda loc, scale: tfd.Mixture(
  cat=tfd.Categorical(probs=tf.constant([0.97, 0.03], dtype=tf.float64)),
  components=[tfd.Normal(loc, scale*5), tfd.Normal(loc, scale*50)])
tfd = tfp.distributions
mix = tf.convert_to_tensor([5, 10, 90, 0.1, 0.3, 0.3e-1, 1e-2, 0.01], tf.float64)
mix = mix / np.sum(mix)
mixture = tfd.Mixture(
  cat=tfd.Categorical(probs=mix),
  components=[
    tfd.Normal(loc=tf.constant(3., tf.float64), scale=3.5),
    SigShape(loc=tf.constant(3.68, tf.float64), scale=0.2e-2),  # psi(2S)
    SigShape(loc=tf.constant(3.01, tf.float64), scale=0.5e-2),  # J/psi
    tfd.Normal(loc=tf.constant(4.1, tf.float64), scale=1e-1),
    tfd.Exponential(tf.constant(9, tf.float64)),
    tfd.Normal(loc=tf.constant(4.5, tf.float64), scale=1.1e-1),
    tfd.Cauchy(loc=tf.constant(4.2, tf.float64), scale=0.3),
    tfd.Poisson(rate=tf.constant(2.4, tf.float64)),
])


functions = [lambda x: (x+1)/(x**3 + x**2 - 6*x),lambda x: (2*tf.math.sin(x))**7
,lambda x: tf.math.cos(tf.math.exp(x))*tf.math.exp(x),lambda x: 1/(1+2500* x**2)
,lambda x: (-1/x**2) * tf.math.cos(1/x),lambda x: x**2 * tf.math.cosh(x)*tf.math.exp(tf.math.exp(x**2))*tf.math.sinh(x),
lambda x: tf.math.exp(-x) * (x**6),lambda x: 250* tf.math.exp(-15000* x**2),
lambda x: tf.math.exp(tf.math.cos(tf.math.exp(x)))-x*tf.math.exp(tf.math.cos(tf.math.exp(x))+x)*tf.math.sin(tf.math.exp(x))
    ,lambda x: mixture.prob(x)]

sol_func = [lambda x: ((-4*tf.math.log(tf.abs(x+3)) -5*tf.math.log(tf.abs(x))
 +9*tf.math.log(tf.abs(x-2)))/30),lambda x: 128*((tf.math.cos(x)**7)/7 -  3*(tf.math.cos(x)**5)/5
+ tf.math.cos(x)**3 - tf.math.cos(x)),lambda x: tf.math.sin(tf.math.exp(x)), lambda x: tf.math.atan(50*x)/50 , lambda x:
tf.math.sin(1/x),lambda x:tf.cast(0,dtype=tf.float64), lambda x: tf.where(x>0,tf.cast(720,dtype=tf.float64),tf.cast(0,dtype=tf.float64)),
lambda x:tf.where(x>0,250*tf.math.sqrt(tf.cast(pi/15000,dtype=tf.float64)),0),
lambda x: x*tf.math.exp(tf.math.cos(tf.math.exp(x))),lambda x:tf.where(x>2.5,mixture.cdf(x),mixture.cdf(x))]


low_lims = tf.convert_to_tensor([0.01,0,5,-1,-0.5,-1,0,-0.04,1,0.1],dtype=tf.float64)
up_lims = tf.convert_to_tensor([1.99,pi,9,1,-0.02,1,35,0.04,7,5.0],dtype=tf.float64)

n=10
integrand = functions[n-1]
tol_array = [1e-4,5e-4,1e-5,5e-5,1e-6,5e-6,1e-7,5e-7,1e-8,5e-8,1e-9,5e-9,1e-10,5e-10,1e-11,5e-11,1e-12]
tol_array.sort(reverse=True)
tol_array = tf.convert_to_tensor(tol_array, dtype=tf.float64)
answer = sol_func[n - 1](up_lims[n - 1]) - sol_func[n - 1](low_lims[n - 1])
print(f"function number {n}")
for i in range(len(tol_array)):
    q = romberg(low_lims[n-1],up_lims[n-1],tol_array[i],5)
    print(tf.abs(q[0]-answer).numpy())
    iter_table = []
    for j in range(30):
        iter_vals = []
        start = time.time()
        a = romberg(low_lims[n-1],up_lims[n-1], tol_array[i],5)
        stop = time.time()
        err = tf.abs(a[0] - answer).numpy()
        timing = stop - start
        iter_vals.append(a[0].numpy())
        iter_vals.append(timing)
        iter_vals.append(a[1].numpy())
        iter_vals.append(err)
        iter_vals.append(tol_array[i])
        iter_table.append(iter_vals)
    average = [(tf.reduce_sum(col).numpy()/len(col)) for col in zip(*iter_table)]
    with open(f'Romberg simpson {n} raw difficult function.csv', 'a', newline='')   as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(iter_table)
    with open(f'Romberg simpson {n} avg difficult function.csv', 'a', newline='')   as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(average)
    with open(f'Romberg simpson CPU and RAM usage {n} difficult function.txt','a',newline='') as file:
        file.write(f'CPU percent = {str(ps.cpu_percent())} \r\n')
        file.write(f'RAM usage = {str(ps.virtual_memory())}\r\n')

'''tol_array = [1e-4,5e-4,1e-5,5e-5,1e-6,5e-6,1e-7,5e-7,1e-8,5e-8,1e-9,5e-9,1e-10,5e-10,1e-11,5e-11,1e-12]
tol_array.sort(reverse=True)
tol_array = tf.convert_to_tensor(tol_array,dtype=tf.float64)
err_array = []
iter_array = []
ratio2 = []
answer_func = lambda x: -tf.math.cos(x)
answer = answer_func(tf.cast(1,dtype=tf.float64))-answer_func(tf.cast(0,dtype=tf.float64))
for i in range(len(tol_array)):
    q = romberg(0,1,tol_array[i])
    error = tf.abs(q[0]-answer)
    err_array.append(error.numpy())
    iter_array.append(q[1].numpy())
    ratio2.append(error.numpy()/tol_array[i].numpy())

print(err_array)
print(tf.reduce_mean(ratio2[0:4]))
print(ratio2[0:4])
print(tf.reduce_mean(ratio2[4:8]))
print(ratio2[4:8])
print(tf.reduce_mean(ratio2[8:]))
print(ratio2[8:])
print(iter_array)'''





