import numpy as np
import tensorflow as tf

EPSILON = 1e-7


def leaky_relu(x):
    return tf.maximum(x, x * 0.2)


def token_id_to_vocab(token_id, vocab):
    token_id = int(np.asscalar(token_id))
    # print("Token: {}, {}".format(token_id, type(token_id)))
    if token_id == 0:
        return '_END_'
    elif token_id == 1:
        return '_UNK_'
    else:
        v = vocab[token_id - 2].decode('ascii')
        # print("v: {},{}".format(v, type(v)))
        return v


def get_kl_weight(params):
    kl_0 = params.kl_0
    rate = 1. / kl_0
    steps = params.kl_anneal_steps
    kl_raw = tf.train.exponential_decay(
        kl_0,
        decay_rate=rate,
        decay_steps=steps,
        global_step=tf.train.get_global_step(),
        name='kl_raw',
        staircase=False)
    kl_weight = tf.minimum(kl_raw, 1., name='kl_weight')
    tf.summary.scalar('kl_weight', kl_weight)
    return kl_weight
