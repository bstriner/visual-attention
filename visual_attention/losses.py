import tensorflow as tf

from .gumbel import softmax_nd
from .util import EPSILON


def cross_entropy_loss(labels, logits, smoothing=0.):
    depth = tf.shape(logits)[-1]
    p = softmax_nd(logits, axis=2)  # (n, depth, vocab+2) [end, unknown] + vocab
    onehot = tf.one_hot(tf.nn.relu(labels - 1), depth, axis=2)  # (n, depth, vocab)
    if smoothing > 0:
        onehot = (onehot * (1. - smoothing)) + (smoothing * tf.ones_like(onehot) / depth)
    mask = 1. - tf.cast(tf.equal(labels, 0), tf.float32)  # (n, depth)
    loss_partial = -(onehot * tf.log(EPSILON + p)) - ((1. - onehot) * tf.log(EPSILON + 1. - p))  # (n, depth, vocab)
    loss_partial = tf.reduce_sum(loss_partial, axis=2)  # (n, depth)
    normalizer = tf.reduce_sum(mask, axis=1) + EPSILON
    loss = tf.reduce_sum(mask * loss_partial, axis=1) / normalizer
    return loss


def nll_loss(labels, logits):
    depth = tf.shape(logits)[-1]
    p = softmax_nd(logits, axis=2)  # (n, depth, vocab+2) [end, unknown] + vocab
    onehot = tf.one_hot(tf.nn.relu(labels - 1), depth, axis=2)  # (n, depth, vocab)
    mask = 1. - tf.cast(tf.equal(labels, 0), tf.float32)  # (n, depth)
    loss_partial = -(onehot * tf.log(EPSILON + p))  # (n, depth, vocab)
    loss_partial = tf.reduce_sum(loss_partial, axis=2)  # (n, depth)
    normalizer = tf.reduce_sum(mask, axis=1)  # + EPSILON
    loss = tf.reduce_sum(mask * loss_partial, axis=1) / normalizer
    return loss
