import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

from .gumbel import gumbel_softmax, gumbel_sigmoid, gumbel_argmax

EPSILON = 1e-7


class BaseStepCell(RNNCell):
    def __init__(self, sen, temperature, img_ctx, params, mode, reuse=None, name=None):
        super(BaseStepCell, self).__init__(_reuse=reuse, name=name)
        self.sen = sen
        self._num_units = params.units
        self.vocab_size = params.vocab_size
        self.frame_size = params.frame_size
        self.temperature = temperature
        self.img_ctx = img_ctx
        self.mode = mode
        scale = 0.05
        self.initializer = tf.initializers.random_uniform(minval=-scale, maxval=scale)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self.vocab_size + 2

    def calc_step(self, y0, h0):
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
        sent_logits = tf.layers.dense(h, units=1, kernel_initializer=self.initializer, name='input_sent')
        with tf.name_scope('recurrent_slot_attention'):
            if self.mode == tf.estimator.ModeKeys.PREDICT:
                slot_attn = tf.one_hot(tf.argmax(input=attn_logits + attn_bias, axis=-1),
                                       tf.shape(attn_logits)[-1], axis=-1)
            else:
                slot_attn = gumbel_softmax(logits=attn_logits + attn_bias, temperature=self.temperature, axis=-1)
            # (n, frames)
        with tf.name_scope('recurrent_slot_sentinel'):
            if self.mode == tf.estimator.ModeKeys.PREDICT:
                slot_sentinel = tf.cast(tf.greater(tf.squeeze(sent_logits, 1), 0), tf.float32)
            else:
                slot_sentinel = gumbel_sigmoid(logits=tf.squeeze(sent_logits, 1), temperature=self.temperature)

        # Calculate forward
        slot_combined_attn = tf.expand_dims(slot_sentinel, 1) * slot_attn
        slot_data = tf.reduce_sum(self.img_ctx * tf.expand_dims(slot_combined_attn, axis=2), axis=1)
        # (n, channels)
        slot_ctx = tf.layers.dense(slot_data, units=self._num_units, kernel_initializer=self.initializer,
                                   name='slot_ctx')
        slot_embed = tf.get_variable(name='slot_embed', shape=[self.frame_size, self._num_units], trainable=True,
                                     initializer=self.initializer)
        slot_embedded = tf.matmul(slot_combined_attn, slot_embed)
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

        return (hlogits, slot_attn, tf.expand_dims(slot_sentinel, 1)), h1


class TrainStepCell(BaseStepCell):
    def __init__(self, sen, temperature, img_ctx, params, mode, reuse=None, name=None):
        super(TrainStepCell, self).__init__(sen=sen, temperature=temperature, mode=mode,
                                            name=name,
                                            img_ctx=img_ctx, params=params, reuse=reuse)

    def call(self, inputs, state):
        y0 = tf.squeeze(inputs, axis=1)  # (n,) [end, unknown]+vocab
        h0 = state
        (hlogits, slot_attn, slot_sentinel), h1 = self.calc_step(y0=y0, h0=h0)
        return (hlogits, slot_attn, slot_sentinel), h1

    @property
    def output_size(self):
        return (self.vocab_size + 2, self.frame_size, 1)


class PredictStepCell(BaseStepCell):
    def __init__(self, sen, img_ctx, params, mode, reuse=None, name=None):
        super(PredictStepCell, self).__init__(sen=sen, temperature=None, name=name,
                                              img_ctx=img_ctx, params=params, reuse=reuse, mode=mode)

    def call(self, inputs, state):
        y0, h0 = state
        y0 = tf.cast(tf.squeeze(y0, axis=1), tf.int32)
        (hlogits, slot_attn, slot_sentinel), h1 = self.calc_step(y0=y0, h0=h0)
        # hlogits: (n, vocab+2) [end, unk]+vocab
        y1 = tf.expand_dims(tf.cast(gumbel_argmax(hlogits, axis=1), tf.float32), 1)  # [pad, start, end, unk] + vocab
        return (hlogits, slot_attn, slot_sentinel, y1), (y1 + 2, h1)

    @property
    def output_size(self):
        return (self.vocab_size + 2, self.frame_size, 1, 1)
