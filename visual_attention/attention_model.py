import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, apply_regularization
from tensorflow.python.ops.rnn_cell import RNNCell

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
        attn = gumbel_softmax(logits=h, temperature=temperature, axis=(1, 2))
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


class StepCell(RNNCell):
    def __init__(self, sen, temperature, img_ctx, params, reuse=None):
        super(StepCell, self).__init__(_reuse=reuse)
        self.sen = sen
        self._num_units = params.units
        self.vocab_size = params.vocab_size
        self.frame_size = params.frame_size
        self.temperature = temperature
        self.img_ctx = img_ctx
        scale = 0.05
        self.initializer = tf.initializers.random_uniform(minval=-scale, maxval=scale)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self.vocab_size + 2

    def call(self, inputs, state):
        y0 = tf.squeeze(inputs, axis=1)  # (n,) [end, unknown]+vocab
        h0 = state
        y_embed = tf.get_variable(name='y_embed', shape=[self.vocab_size + 4, self._num_units], trainable=True,
                                  initializer=self.initializer)
        y_embedded = tf.gather(y_embed, y0, axis=0)  # n, unit

        sen_ctx0 = tf.layers.dense(self.sen, units=self._num_units, kernel_initializer=self.initializer,
                                   name='sen_ctx0')
        sen_ctx1 = tf.layers.dense(1. - self.sen, units=self._num_units, kernel_initializer=self.initializer,
                                   name='sen_ctx1')

        # Select input slot
        inp = h0
        inp += sen_ctx0
        inp += sen_ctx1
        inp += y_embedded
        h = inp
        for j in range(3):
            h = tf.layers.dense(h, units=self._num_units, kernel_initializer=self.initializer,
                                name='input_attn{}'.format(j))
            h = tf.nn.relu(h)
        attn_logits = tf.layers.dense(h, units=self.frame_size, kernel_initializer=self.initializer,
                                      name='input_attn_final')
        attn_bias = tf.log(1e-5 + self.sen)
        with tf.name_scope('slot_attention_gumbel'):
            slot_attn = gumbel_softmax(logits=attn_logits + attn_bias, temperature=self.temperature, axis=-1)
            # (n, frames)

        # Calculate forward
        slot_data = tf.reduce_sum(self.img_ctx * tf.expand_dims(slot_attn, axis=2), axis=1)  # (n, channels)
        slot_ctx = tf.layers.dense(slot_data, units=self._num_units, kernel_initializer=self.initializer,
                                   name='slot_ctx')
        slot_embed = tf.get_variable(name='slot_embed', shape=[self.frame_size, self._num_units], trainable=True,
                                     initializer=self.initializer)
        slot_embedded = tf.matmul(slot_attn, slot_embed)
        inp = h0
        inp += sen_ctx0
        inp += sen_ctx1
        inp += y_embedded
        inp += slot_ctx
        inp += slot_embedded
        h = inp
        for j in range(3):
            h = tf.layers.dense(h, units=self._num_units, kernel_initializer=self.initializer,
                                name='forward{}'.format(j))
            h = tf.nn.relu(h)
        hd = tf.layers.dense(h, units=self._num_units, kernel_initializer=self.initializer,
                             name='forwardfinal')
        h1 = h0 + hd
        hlogits = tf.layers.dense(h, units=self.vocab_size + 2, kernel_initializer=self.initializer,
                                  name='forwardlogits')

        return hlogits, h1


def decoder_fn(img_ctx, sen, cap, temperature, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError
    elif mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
        shifted = shift_captions(cap)
        shifted = tf.expand_dims(shifted, axis=2)
        shape = tf.shape(cap)
        n = shape[0]
        cell = StepCell(sen=sen, temperature=temperature, img_ctx=img_ctx, params=params)
        initial_state = cell.zero_state(batch_size=n, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=shifted, initial_state=initial_state, time_major=False)
        return outputs  # (n, depth, vocab)
    else:
        raise ValueError()


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


def model_fn(features, labels, mode, params):
    img = features['images']  # (image_n,h, w, c)
    cap = features['captions']  # (caption_n, depth)
    ass = features['assignments']  # (caption_n,)

    temperature = tf.train.exponential_decay(params.tau_0,
                                             decay_rate=params.tau_decay_rate,
                                             decay_steps=params.tau_decay_steps,
                                             global_step=tf.train.get_global_step(),
                                             name='temperature',
                                             staircase=False)
    tf.summary.scalar('temperature', temperature)

    attn, sen = attention_fn(img, temperature=temperature, mode=mode, params=params)
    ctx = apply_attn(img=img, att=attn, sen=sen)  # (image_n, frames, c)
    decoder_ctx = tf.gather(ctx, ass, axis=0)  # (caption_n, frames, c)
    decoder_sen = tf.gather(sen, ass, axis=0)  # (caption_n, frames)

    logits = decoder_fn(img_ctx=decoder_ctx, sen=decoder_sen, cap=cap,
                        temperature=temperature, mode=mode, params=params)
    classes = tf.argmax(logits, axis=2)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": classes
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        loss = tf.reduce_mean(cross_entropy(labels=cap, logits=logits, vocab_size=params.vocab_size))
        if params.l2 > 0:
            reg = apply_regularization(l2_regularizer(params.l2), tf.trainable_variables())
            tf.summary.scalar("regularization", reg)
            loss += reg
        if params.sen_l1 > 0:
            sen_reg = params.sen_l1 * tf.reduce_mean(tf.reduce_sum(sen, axis=1), axis=0)
            tf.summary.scalar('sentinel_regularization', sen_reg)
            loss += sen_reg
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
