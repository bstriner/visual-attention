import tensorflow as tf

from .gumbel import gumbel_softmax, gumbel_sigmoid, softmax_nd
from .util import leaky_relu


def attention_fn(img, temperature, mode, params):
    n = tf.shape(img)[0]
    activation = leaky_relu
    cnn_args = {}
    training = mode == tf.estimator.ModeKeys.TRAIN
    frame_size = params.frame_size

    # Convolutional network
    h = img
    for i in range(3):
        h = tf.layers.conv2d(inputs=h, filters=params.units, kernel_size=[3, 3],
                             padding="same", name='attn_conv{}'.format(i), **cnn_args)
        h = activation(h)

    h_att = tf.layers.conv2d(inputs=h, filters=frame_size, kernel_size=[3, 3],
                             padding="same", name='attn_att', **cnn_args)
    h_sen = tf.layers.conv2d(inputs=h, filters=frame_size, kernel_size=[3, 3],
                             padding="same", name='attn_sen', **cnn_args)

    # attention
    if params.attn_mode_img == 'gumbel':
        # h = tf.transpose(h_att, (0, 3, 1, 2))  # (n,c, h,w)
        # h = tf.reshape(h, (-1, 14 * 14))  # (n*c, h*w)
        # h = gumbel_softmax(logits=h, temperature=temperature, axis=-1)
        # h = tf.reshape(h, (-1, frame_size, 14, 14))  # (n,c, h, w)
        # attn = tf.transpose(h, (0, 2, 3, 1))  # (n, h, w, c)
        h = tf.transpose(h_att, (0, 3, 1, 2))  # (n,c,h,w)
        h = tf.reshape(h, (n * frame_size, 14 * 14))  # (n*c, h*w)
        if mode == tf.estimator.ModeKeys.PREDICT:
            h = tf.one_hot(tf.argmax(h, axis=1), tf.shape(h)[1], axis=1)
        else:
            h = gumbel_softmax(logits=h, temperature=temperature, axis=1)
        h = tf.reshape(h, (n, frame_size, 14, 14))
        attn = tf.transpose(h, (0, 2, 3, 1))
    elif params.attn_mode_img == 'soft':
        attn = softmax_nd(h_att, axis=(1, 2))
    else:
        raise ValueError()

    # sentinel
    h = tf.reduce_mean(h_sen, axis=(1, 2))  # (n, c)
    sen = gumbel_sigmoid(h, temperature=temperature)  # (n, c)
    tf.summary.histogram('image_sentinel', sen)
    return attn, sen


def apply_attn(img, att):
    # img (n, w, h, c)
    # att (n, w, h, frames)
    # sen (n, frames)
    h = tf.expand_dims(img, axis=3) * tf.expand_dims(att, axis=4)  # (n, w, h, frames, c)
    h = tf.reduce_sum(h, axis=(1, 2))  # (n, frames, c)
    return h  # (n, frames, c)


def slot_vocab_fn(img_ctx, params):
    # img_ctx: (n, frames, c)
    h = img_ctx
    for i in range(3):
        h = tf.layers.dense(inputs=h, units=params.units, name='slot_vocab_{}'.format(i))
        h = leaky_relu(h)
    vocab = tf.layers.dense(inputs=h, units=params.vocab_size + 1, name='slot_vocab_logits')
    return vocab  # (n, frames, vocab+1)
