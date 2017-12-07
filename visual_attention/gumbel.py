import tensorflow as tf


def softmax_nd(x, axis=-1):
    e_x = tf.exp(x - tf.reduce_max(x, axis=axis, keep_dims=True))
    out = e_x / tf.reduce_sum(e_x, axis=axis, keep_dims=True)
    return out


def sample_gumbel(shape, eps=1e-9):
    rnd = tf.random_uniform(shape=shape, minval=eps, maxval=1. - eps, dtype='float32')
    return -tf.log(eps - tf.log(eps + rnd))


def gumbel_softmax_sample(logits, temperature, axis=-1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return softmax_nd(y / temperature, axis=axis)


def gumbel_sigmoid(logits, temperature, hard=False):
    #g1 = sample_gumbel(logit.shape)
    #g2 = sample_gumbel(logit.shape)
    #a = tf.exp((g1 + logit) / temperature)
    #b = tf.exp((g2 - logit) / temperature)
    #s = a + b
    #todo: rescale subtract max
    #return a / s
    return tf.gather(gumbel_softmax(tf.stack((-logits, logits), axis=0),
                                    temperature=temperature, hard=hard, axis=0), 1, axis=0)


def gumbel_argmax(logits, axis=-1):
    g = sample_gumbel(shape=tf.shape(logits))
    return tf.argmax(logits + g, axis=axis)


def sample_one_hot(logits, srng, axis=-1):
    g = sample_gumbel(shape=logits.shape)
    h = logits + g
    return tf.one_hot(tf.argmax(h, axis=axis), tf.shape(h)[axis])


def gumbel_softmax(logits, temperature, hard=False, axis=-1):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, axis=axis)
    if hard:
        y_hard = tf.one_hot(tf.argmax(y, axis=axis), tf.shape(y)[axis], axis=axis)
        y = tf.stop_gradient(y_hard - y) + y
    return y
