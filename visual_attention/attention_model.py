import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, apply_regularization

from .attention_step import PredictStepCell, TrainStepCell
from .gumbel import gumbel_softmax, gumbel_sigmoid, softmax_nd

EPSILON = 1e-7


def attention_fn(img, temperature, mode, params):
    activation = tf.nn.relu
    cnn_args = {}
    # if params.kernel_l2 > 0:
    #    cnn_args['kernel_regularizer'] = l2(params.kernel_l2)
    # if params.bias_l2 > 0:
    #    cnn_args['bias_regularizer'] = l2(params.bias_l2)
    training = mode == tf.estimator.ModeKeys.TRAIN
    frame_size = params.frame_size
    h = img
    h = tf.layers.conv2d(inputs=h, filters=256, kernel_size=[3, 3],
                         padding="same", name='attn_conv1', **cnn_args)
    h = activation(h)
    h = tf.layers.conv2d(inputs=h, filters=256, kernel_size=[3, 3],
                         padding="same", name='attn_conv2', **cnn_args)
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
        h = tf.transpose(h, (0, 3, 1, 2))  # (n,c,h,w)
        h = tf.reshape(h, (-1, 14 * 14))  # (n*c, h*w)
        if mode == tf.estimator.ModeKeys.PREDICT:
            h = tf.one_hot(tf.argmax(h, axis=1), tf.shape(h)[1], axis=1)
        else:
            h = gumbel_softmax(logits=h, temperature=temperature, axis=1)
        h = tf.reshape(h, (-1, frame_size, 14, 14))
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


def shift_captions(cap):
    n = tf.shape(cap)[0]
    zeros = tf.zeros((n, 1), dtype=tf.int32)
    shifted = tf.concat((zeros, cap[:, :-1] + 1), axis=1)
    return shifted


def predict_decoder_fn(img_ctx, sen, params, mode, depth):
    shape = tf.shape(img_ctx)
    n = shape[0]
    y0 = tf.zeros((n, 1), dtype=tf.float32)
    cell = PredictStepCell(sen=sen, img_ctx=img_ctx, params=params, mode=mode, name='step_cell')
    initial_state = cell.zero_state(batch_size=n, dtype=tf.float32)
    inputs = tf.zeros((1, depth, 1))
    (logits, slot_attn, slot_sentinel, y1), state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=inputs,
        initial_state=(y0, initial_state),
        time_major=False)
    return logits, slot_attn, slot_sentinel, y1


def train_decoder_fn(img_ctx, sen, cap, temperature, params, mode):
    shape = tf.shape(img_ctx)
    n = shape[0]
    shifted = shift_captions(cap)
    shifted = tf.expand_dims(shifted, axis=2)
    cell = TrainStepCell(sen=sen, temperature=temperature, img_ctx=img_ctx, params=params, mode=mode, name='step_cell')
    initial_state = cell.zero_state(batch_size=n, dtype=tf.float32)
    (hlogits, slot_attn, slot_sentinel), h1 = tf.nn.dynamic_rnn(cell=cell, inputs=shifted,
                                                                initial_state=initial_state, time_major=False)
    return hlogits, slot_attn, slot_sentinel  # (n, depth, vocab)


def apply_attn(img, att, sen):
    # img (n, w, h, c)
    # att (n, w, h, frames)
    h = tf.expand_dims(img, axis=3) * tf.expand_dims(att, axis=4)  # (n, w, h, frames, c)
    h = tf.reduce_sum(h, axis=(1, 2))  # (n, frames, c)
    h *= tf.expand_dims(sen, axis=2)  # (n, frames, c)
    return h


def cross_entropy(labels, logits, vocab_size):
    p = softmax_nd(logits, axis=2)  # (n, depth, vocab+2) [end, unknown] + vocab
    onehot = tf.one_hot(tf.nn.relu(labels - 1), vocab_size + 2, axis=2)  # (n, depth, vocab)
    mask = 1. - tf.cast(tf.equal(labels, 0), tf.float32)  # (n, depth)
    loss_partial = -(onehot * tf.log(EPSILON + p)) - ((1. - onehot) * tf.log(EPSILON + 1. - p))  # (n, depth, vocab)
    loss_partial = tf.reduce_sum(loss_partial, axis=2)  # (n, depth)
    normalizer = tf.reduce_sum(mask, axis=1) + EPSILON
    loss = tf.reduce_sum(mask * loss_partial, axis=1) / normalizer
    return loss


def nll_loss(labels, logits, vocab_size):
    p = softmax_nd(logits, axis=2)  # (n, depth, vocab+2) [end, unknown] + vocab
    onehot = tf.one_hot(tf.nn.relu(labels - 1), vocab_size + 2, axis=2)  # (n, depth, vocab)
    mask = 1. - tf.cast(tf.equal(labels, 0), tf.float32)  # (n, depth)
    loss_partial = -(onehot * tf.log(EPSILON + p))  # (n, depth, vocab)
    loss_partial = tf.reduce_sum(loss_partial, axis=2)  # (n, depth)
    normalizer = tf.reduce_sum(mask, axis=1)  # + EPSILON
    loss = tf.reduce_sum(mask * loss_partial, axis=1) / normalizer
    return loss


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


def model_fn(features, labels, mode, params):
    img = features['images']  # (image_n,h, w, c)

    temperature = get_temperature(params)
    img_attn, img_sen = attention_fn(img, temperature=temperature, mode=mode, params=params)
    img_ctx = apply_attn(img=img, att=img_attn, sen=img_sen)  # (image_n, frames, c)

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits, slot_attn, slot_sentinel, y1 = predict_decoder_fn(
            img_ctx=img_ctx, sen=img_sen, params=params, depth=30, mode=mode)
        predictions = {
            'classes': y1,
            'image_ids': tf.get_default_graph().get_tensor_by_name('image_ids:0'),
            'slot_attention': slot_attn,
            'slot_sentinel': slot_sentinel,
            'image_attention': img_attn,
            'image_sentinel': img_sen
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        cap = features['captions']  # (caption_n, depth)
        ass = features['assignments']  # (caption_n,)
        decoder_ctx = tf.gather(img_ctx, ass, axis=0)  # (caption_n, frames, c)
        decoder_sen = tf.gather(img_sen, ass, axis=0)  # (caption_n, frames)

        logits, slot_attn, slot_sentinel = train_decoder_fn(img_ctx=decoder_ctx, sen=decoder_sen, cap=cap,
                                                            temperature=temperature, params=params, mode=mode)
        loss = tf.reduce_mean(nll_loss(labels=cap, logits=logits, vocab_size=params.vocab_size))
        if params.l2 > 0:
            reg = apply_regularization(l2_regularizer(params.l2), tf.trainable_variables())
            tf.summary.scalar("regularization", reg)
            loss += reg
        if params.img_sen_l1 > 0:
            img_sen_reg = params.img_sen_l1 * tf.reduce_mean(tf.reduce_sum(img_sen, axis=1), axis=0)
            tf.summary.scalar('image_sentinel_regularization', img_sen_reg)
            loss += img_sen_reg
        if params.slot_sen_l1 > 0:
            slot_sen_reg = params.slot_sen_l1 * tf.reduce_mean(tf.reduce_sum(slot_sentinel, axis=(1, 2)), axis=0)
            tf.summary.scalar('slot_sentinel_regularization', slot_sen_reg)
            loss += slot_sen_reg
        if mode == tf.estimator.ModeKeys.TRAIN:
            lr = tf.train.exponential_decay(params.lr,
                                            decay_rate=params.decay_rate,
                                            decay_steps=params.decay_steps,
                                            global_step=tf.train.get_global_step(),
                                            name='learning_rate',
                                            staircase=False)
            tf.summary.scalar('learning_rate', lr)
            if params.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            elif params.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=params.momentum)
            elif params.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=params.momentum)
            else:
                raise ValueError("Unknown optimizer: {}".format(params.optimizer))
            print("Trainable: {}".format(list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))))
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        else:
            eval_metric_ops = {}
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
