import tensorflow as tf

from .gumbel import softmax_nd
from .util import EPSILON


def cross_entropy_loss(labels, logits, mask=None, smoothing=0.):
    depth = tf.shape(logits)[-1]
    p = softmax_nd(logits, axis=2)  # (n, depth, vocab+2) [end, unknown] + vocab
    onehot = tf.one_hot(labels, depth, axis=2)  # (n, depth, vocab)
    if smoothing > 0:
        onehot = (onehot * (1. - smoothing)) + (smoothing * tf.ones_like(onehot) / depth)
    loss_partial = -(onehot * tf.log(EPSILON + p)) - ((1. - onehot) * tf.log(EPSILON + 1. - p))  # (n, depth, vocab)
    loss_partial = tf.reduce_sum(loss_partial, axis=2)  # (n, depth)
    if mask is not None:
        normalizer = tf.reduce_sum(mask, axis=1) + EPSILON
        loss = tf.reduce_sum(mask * loss_partial, axis=1) / normalizer
    else:
        loss = tf.reduce_mean(loss_partial, axis=(1, 2))
    return loss


def nll_loss(labels, logits, mask=None, mean=True):
    depth = tf.shape(logits)[-1]
    p = softmax_nd(logits, axis=2)  # (n, depth, vocab+2) [end, unknown] + vocab
    shape = tf.shape(p)
    mg = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing='ij')
    idx = tf.stack([mg[0], mg[1], labels], axis=2)
    p_t = tf.gather_nd(p, idx)
    loss_partial = -tf.log(EPSILON+p_t)
    #onehot = tf.one_hot(labels, depth, axis=2)  # (n, depth, vocab)
    #loss_partial = -(onehot * tf.log(EPSILON + p))  # (n, depth, vocab)
    #loss_partial = tf.reduce_sum(loss_partial, axis=2)  # (n, depth)
    if mean:
        if mask is not None:
            normalizer = tf.reduce_sum(mask, axis=1)  # + EPSILON
            loss = tf.reduce_sum(mask * loss_partial, axis=1) / normalizer
        else:
            loss = tf.reduce_mean(loss_partial, axis=1)
    else:
        if mask is not None:
            loss = tf.reduce_sum(mask * loss_partial, axis=1)
        else:
            loss = tf.reduce_sum(loss_partial, axis=1)
    return loss
