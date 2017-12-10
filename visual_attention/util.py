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
