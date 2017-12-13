import tensorflow as tf


def modal_sample_softmax(logit, temperature, mode, attn_mode='gumbel', axis=-1):
    if attn_mode == 'soft':
        return softmax_nd(logit, axis=axis)
    elif attn_mode == 'gumbel':
        if mode == tf.estimator.ModeKeys.PREDICT:
            if tf.flags.FLAGS.deterministic:
                #return take_one_hot(logit, axis=axis)
                print("TEMP:{}".format(temperature))
                return softmax_nd(logit/temperature, axis=axis)
            else:
                #return sample_one_hot(logits=logit, axis=axis)
                return gumbel_softmax(logits=logit, axis=axis, temperature=temperature)
        else:
            return gumbel_softmax(logits=logit, axis=axis, temperature=temperature)
    else:
        raise ValueError()


def modal_sample_sigmoid(logit, temperature, mode, attn_mode='gumbel'):
    if attn_mode == 'soft':
        return tf.nn.sigmoid(logit)
    elif attn_mode == 'gumbel':
        if mode == tf.estimator.ModeKeys.PREDICT:
            if tf.flags.FLAGS.deterministic:
                #return tf.cast(tf.greater(logit, 0), tf.float32)
                return tf.nn.sigmoid(logit/temperature)
            else:
                #return sample_sigmoid(logits=logit)
                return gumbel_sigmoid(logits=logit, temperature=temperature)
        else:
            return gumbel_sigmoid(logits=logit, temperature=temperature)
    else:
        raise ValueError()


def get_temperature(params):
    temperature_raw = tf.train.exponential_decay(params.tau_0,
                                                 decay_rate=params.tau_decay_rate,
                                                 decay_steps=params.tau_decay_steps,
                                                 global_step=tf.train.get_global_step(),
                                                 name='temperature_raw',
                                                 staircase=False)
    temperature = tf.maximum(temperature_raw, params.tau_min, name='temperature')
    tf.summary.scalar('temperature', temperature)
    return temperature


def take_one_hot(x, axis=-1, depth=None):
    depth = best_shape(x, axis=axis, depth=depth)
    ret = tf.one_hot(tf.argmax(x, axis=axis), depth=depth, axis=axis)
    return ret


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
    # g1 = sample_gumbel(logit.shape)
    # g2 = sample_gumbel(logit.shape)
    # a = tf.exp((g1 + logit) / temperature)
    # b = tf.exp((g2 - logit) / temperature)
    # s = a + b
    # todo: rescale subtract max
    # return a / s
    return tf.gather(gumbel_softmax(tf.stack((-logits, logits), axis=0),
                                    temperature=temperature, hard=hard, axis=0), 1, axis=0)


def sample_argmax(logits, axis=-1):
    g = sample_gumbel(shape=tf.shape(logits))
    return tf.argmax(logits + g, axis=axis)


def best_shape(x, axis, depth=None):
    if depth is None:
        depth = x.get_shape()[axis].value
    if depth is None:
        depth = tf.shape(x)[axis]
    return depth


def sample_one_hot(logits, axis=-1, depth=None):
    depth = best_shape(logits, axis=axis, depth=depth)
    g = sample_gumbel(shape=tf.shape(logits))
    h = logits + g
    return tf.one_hot(tf.argmax(h, axis=axis), depth=depth, axis=axis)


def sample_sigmoid(logits):
    sig = tf.nn.sigmoid(logits)
    rnd = tf.random_uniform(shape=tf.shape(sig), minval=0, maxval=1, dtype=tf.float32)
    samp = tf.cast(tf.greater(sig, rnd), tf.float32)
    return samp


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
