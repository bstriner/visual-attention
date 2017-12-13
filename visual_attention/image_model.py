import tensorflow as tf

from .gumbel import modal_sample_sigmoid, modal_sample_softmax
from .util import leaky_relu


def attention_fn(img, temperature, mode, params):
    n = tf.shape(img)[0]
    activation = leaky_relu
    cnn_args = {}
    training = mode == tf.estimator.ModeKeys.TRAIN
    frame_size = params.frame_size

    # Convolutional network
    h = img
    if params.dropout_img_input > 0:
        h = tf.layers.dropout(inputs=h, rate=params.dropout_img_input, training=training)
    for i in range(params.depth):
        h = tf.layers.conv2d(inputs=h, filters=params.units, kernel_size=[3, 3],
                             padding="same", name='attn_conv{}'.format(i), **cnn_args)
        h = activation(h)
        if params.dropout_img_hidden > 0:
            h = tf.layers.dropout(inputs=h, rate=params.dropout_img_hidden, training=training)

    h_att = tf.layers.conv2d(inputs=h, filters=frame_size, kernel_size=[3, 3],
                             padding="same", name='attn_att', **cnn_args)
    h_sen = tf.layers.conv2d(inputs=h, filters=frame_size, kernel_size=[3, 3],
                             padding="same", name='attn_sen', **cnn_args)

    # attention
    h = tf.transpose(h_att, (0, 3, 1, 2))  # (n,c,h,w)
    h = tf.reshape(h, (n * frame_size, 14 * 14))  # (n*c, h*w)
    h = modal_sample_softmax(
        logit=h,
        temperature=temperature,
        mode=mode,
        attn_mode=params.attn_mode_img,
        axis=1)
    h = tf.reshape(h, (n, frame_size, 14, 14))
    attn = tf.transpose(h, (0, 2, 3, 1))

    if params.use_img_sen:
        # sentinel
        sen_logits = tf.reduce_mean(h_sen, axis=(1, 2))  # (n, c)
        sen = modal_sample_sigmoid(
            logit=sen_logits,
            temperature=temperature,
            mode=mode,
            attn_mode=params.attn_mode_sen)
        tf.summary.histogram('image_sentinel', sen)
        return attn, sen
    else:
        return attn, None


def apply_attn(img, att, sen=None):
    # img (n, w, h, c)
    # att (n, w, h, frames)
    # sen (n, frames)
    h = tf.expand_dims(img, axis=3) * tf.expand_dims(att, axis=4)  # (n, w, h, frames, c)
    h = tf.reduce_sum(h, axis=(1, 2))  # (n, frames, c)
    if sen is not None:
        h *= tf.expand_dims(sen, axis=2)
    return h  # (n, frames, c)


def slot_vocab_fn(img_ctx, params):
    # img_ctx: (n, frames, c)
    initializer = tf.initializers.random_uniform(-0.05, 0.05)
    slot_vocab_embedding = tf.get_variable(
        name='slot_vocab_embedding',
        shape=[params.frame_size, params.units],  # [end, unknown] + vocab
        trainable=True,
        initializer=initializer)
    img_embedded = tf.layers.dense(
        inputs=img_ctx,
        units=params.units,
        name='img_vocab_embed',
        kernel_initializer=initializer)
    h = leaky_relu(img_embedded + tf.expand_dims(slot_vocab_embedding, 0))
    for i in range(params.depth):
        h = tf.layers.dense(inputs=h, units=params.units, name='slot_vocab_{}'.format(i))
        h = leaky_relu(h)
    vocab = tf.layers.dense(inputs=h, units=params.vocab_size + 1, name='slot_vocab_logits')
    return vocab  # (n, frames, vocab+1)
