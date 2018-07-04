"""
Quick debug script for slicing and assembling kernels in multiple convolutional layers.
"""

import numpy as np
import tensorflow as tf

# constants
_BATCH_SIZE = 1
# representation tensors
_DIM_R_H = 16
_DIM_R_W = 16
_DIM_R_A = 256 # slice A
_DIM_R_S = 263 # slice S
_DIM_R_B = 256 # slice B
# convolutional kernels
_DIM_K_H = 5
_DIM_K_W = 5
_DIM_K_N = 256 # number of filters

# shared part between representations
r_s = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_S], dtype=tf.float32)
# representation p
r_a_p = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A], dtype=tf.float32)
r_b_p = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B], dtype=tf.float32)
r_p = tf.concat([r_a_p, r_s, r_b_p], axis=3)
# representation q
# r_a_q = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A], dtype=tf.float32)
# r_b_q = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B], dtype=tf.float32)
# r_q = tf.concat([r_a_q, r_s, r_b_q], axis=3)

# kernel initializers
dim_r = _DIM_R_A + _DIM_R_S + _DIM_R_B
k_init = np.random.rand(_DIM_K_H, _DIM_K_W, dim_r, _DIM_K_N).astype(np.float32)
k_s_init = k_init[:, :, _DIM_R_A : _DIM_R_A + _DIM_R_S, :]
k_a_p_init = k_init[:, :, 0 : _DIM_R_A, :]
k_b_p_init = k_init[:, :, _DIM_R_A + _DIM_R_S : dim_r, :]

# full kernel
k_f = tf.Variable(k_init)
# shared part between kernels
k_s = tf.Variable(k_s_init)
# kernel parts for representation p
k_a_p = tf.Variable(k_a_p_init)
k_b_p = tf.Variable(k_b_p_init)
# assembled kernel for representation p
k_c_p = tf.concat([k_a_p, k_s, k_b_p], axis=2)

# convolutions
conv_k_f = tf.nn.conv2d(r_p, k_f, [1, 1, 1, 1], 'VALID')
conv_k_c = tf.nn.conv2d(r_p, k_c_p, [1, 1, 1, 1], 'VALID')

# run a session and print results
sess = tf.Session()
sess.run(tf.initialize_all_variables())
# randomize representation
r_s_init = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_S)
r_a_p_init = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A)
r_b_p_init = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B)

o1, o2 = sess.run(
    fetches=[conv_k_f, conv_k_c],
    feed_dict={
      r_s : r_s_init,
      r_a_p : r_a_p_init,
      r_b_p : r_b_p_init,
    }
)

print(o1.shape)
print(o2.shape)
