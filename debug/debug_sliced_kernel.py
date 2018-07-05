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
_DIM_K_N = 1 # number of filters: 256

# shared part between representations
r_s = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_S], dtype=tf.float32)
# representation p
r_a_p = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A], dtype=tf.float32)
r_b_p = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B], dtype=tf.float32)
r_p = tf.concat([r_a_p, r_s, r_b_p], axis=3)
# representation q
r_a_q = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A], dtype=tf.float32)
r_b_q = tf.placeholder(shape=[_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B], dtype=tf.float32)
r_q = tf.concat([r_a_q, r_s, r_b_q], axis=3)

# kernel initializers
dim_r = _DIM_R_A + _DIM_R_S + _DIM_R_B
# k_p_init = np.random.rand(_DIM_K_H, _DIM_K_W, dim_r, _DIM_K_N).astype(np.float32)
k_p_init = np.ones([_DIM_K_H, _DIM_K_W, dim_r, _DIM_K_N], dtype=np.float32)
k_s_init = k_p_init[:, :, _DIM_R_A : _DIM_R_A + _DIM_R_S, :]
k_a_p_init = k_p_init[:, :, 0 : _DIM_R_A, :]
k_b_p_init = k_p_init[:, :, _DIM_R_A + _DIM_R_S : dim_r, :]
# k_q_init = np.random.rand(_DIM_K_H, _DIM_K_W, dim_r, _DIM_K_N).astype(np.float32)
k_q_init = np.ones([_DIM_K_H, _DIM_K_W, dim_r, _DIM_K_N], dtype=np.float32)
k_a_q_init = k_q_init[:, :, 0 : _DIM_R_A, :]
k_b_q_init = k_q_init[:, :, _DIM_R_A + _DIM_R_S : dim_r, :]

# full kernel
k_p = tf.Variable(k_p_init)
# shared part between kernels
k_s = tf.Variable(k_s_init)
# kernel parts for representation p
k_a_p = tf.Variable(k_a_p_init)
k_b_p = tf.Variable(k_b_p_init)
# assembled kernel for representation p
k_c_p = tf.concat([k_a_p, k_s, k_b_p], axis=2)
# kernel parts for representation q
k_a_q = tf.Variable(k_a_q_init)
k_b_q = tf.Variable(k_b_q_init)
# assembled kernel for representation q
k_c_q = tf.concat([k_a_q, k_s, k_b_q], axis=2)

# full convolutions
conv_k_f_p = tf.nn.conv2d(r_p, k_p, [1, 1, 1, 1], 'VALID')
conv_k_c_p = tf.nn.conv2d(r_p, k_c_p, [1, 1, 1, 1], 'VALID')
conv_k_c_q = tf.nn.conv2d(r_q, k_c_q, [1, 1, 1, 1], 'VALID')
# partial convolutions
conv_k_a_p = tf.nn.conv2d(r_a_p, k_a_p, [1, 1, 1, 1], 'VALID')
conv_k_s = tf.nn.conv2d(r_s, k_s, [1, 1, 1, 1], 'VALID')
conv_k_b_p = tf.nn.conv2d(r_b_p, k_b_p, [1, 1, 1, 1], 'VALID')
conv_sum_p = tf.add_n([conv_k_a_p, conv_k_s, conv_k_b_p])

# run a session and print results
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# randomize representation
# r_s_init = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_S)
# r_a_p_init = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A)
# r_b_p_init = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B)
# r_a_q_init = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A)
# r_b_q_init = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B)

scale = -9 * 10e-2
r_s_init = np.ones([_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_S], dtype=np.float32) * scale
r_a_p_init = np.ones([_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A], dtype=np.float32) * scale
r_b_p_init = np.ones([_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B], dtype=np.float32) * scale
r_a_q_init = np.zeros([_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_A], dtype=np.float32) * scale
r_b_q_init = np.zeros([_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_B], dtype=np.float32) * scale

o_kfp, o_sum, o_kcp, o_kcq = sess.run(
    fetches=[conv_k_f_p, conv_sum_p, conv_k_c_p, conv_k_c_q],
    feed_dict={
        r_s : r_s_init,
        r_a_p : r_a_p_init,
        r_b_p : r_b_p_init,
        r_a_q : r_a_q_init,
        r_b_q : r_b_q_init,
    }
)

err1 = np.sum(np.abs(o_kcp - o_kfp))
print(err1)
err2 = np.sum(np.abs(o_sum - o_kfp))
print(err2)
diff = np.sum(np.abs(o_kcp - o_kcq))
print(diff)
