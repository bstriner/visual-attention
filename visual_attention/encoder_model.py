import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

from .gumbel import modal_sample_sigmoid, modal_sample_softmax
from .util import leaky_relu


class EncoderStepCell(RNNCell):
    def __init__(
            self,
            sen,
            temperature,
            slot_vocab,
            img_ctx,
            params,
            mode,
            activation=leaky_relu,
            reuse=None,
            name=None):
        super(RNNCell, self).__init__(_reuse=reuse, name=name)
        self.activation = activation
        self.sen = sen
        self._num_units = params.units
        self.frame_size = params.frame_size
        self.dropout_input = params.encoder_dropout_input
        self.dropout_hidden = params.encoder_dropout_hidden
        self.temperature = temperature
        self.slot_vocab = slot_vocab
        self.img_ctx = img_ctx
        self.mode = mode
        self.training = mode == tf.estimator.ModeKeys.TRAIN
        self.params = params
        scale = 0.05
        self.initializer = tf.initializers.random_uniform(minval=-scale, maxval=scale)

    @property
    def state_size(self):
        return self._num_units

    def build_hidden_state(self, n):
        h0 = tf.get_variable(
            name='encoder_h0',
            shape=[1, self._num_units],
            trainable=True,
            initializer=self.initializer)
        h0_tiled = tf.tile(h0, [n, 1])
        return h0_tiled

    @property
    def output_size(self):
        return self._num_units

    def calc_attn(self, inp):
        h = inp
        if self.dropout_input > 0:
            h = tf.layers.dropout(inputs=h, rate=self.dropout_input, training=self.training)
        for j in range(self.params.depth):
            h = tf.layers.dense(
                inputs=h,
                units=self._num_units,
                kernel_initializer=self.initializer,
                name='encoder_attention_{}'.format(j))
            h = self.activation(h)
            if self.dropout_hidden > 0:
                h = tf.layers.dropout(inputs=h, rate=self.dropout_hidden, training=self.training)
        att_logit = tf.layers.dense(
            inputs=h,
            units=self.frame_size,
            kernel_initializer=self.initializer,
            name='encoder_attention_out')
        att = modal_sample_softmax(att_logit, axis=-1, mode=self.mode, temperature=self.temperature)
        sen_logit = tf.layers.dense(
            inputs=h,
            units=1,
            kernel_initializer=self.initializer,
            name='encoder_attention_sen_out')
        sen = modal_sample_sigmoid(sen_logit, temperature=self.temperature, mode=self.mode)
        combined_att = att * sen
        return combined_att

    def calc_hidden(self, inp):
        h = inp
        if self.dropout_input > 0:
            h = tf.layers.dropout(inputs=h, rate=self.dropout_input, training=self.training)
        for j in range(self.params.depth):
            h = tf.layers.dense(
                inputs=h,
                units=self._num_units,
                kernel_initializer=self.initializer,
                name='encoder_hidden_{}'.format(j))
            h = self.activation(h)
            if self.dropout_hidden > 0:
                h = tf.layers.dropout(inputs=h, rate=self.dropout_hidden, training=self.training)
        hd = tf.layers.dense(
            inputs=h,
            units=self._num_units,
            kernel_initializer=self.initializer,
            name='encoder_hidden_out')
        return hd

    def call(self, inputs, state):
        y0, mask = inputs
        y0 = tf.squeeze(y0, 1)
        h0 = state

        input_embedding = tf.get_variable(
            name='y_embed_encoder',
            shape=[self.params.vocab_size + 2, self._num_units],  # [end, unknown] + vocab
            trainable=True,
            initializer=self.initializer)
        input_embedded = tf.gather(input_embedding, y0)

        # attention
        inp = h0 + input_embedded
        attn = self.calc_attn(inp)
        img_ctx = tf.reduce_sum(self.img_ctx * tf.expand_dims(attn, axis=2), axis=1)

        # forward
        img_embedded = tf.layers.dense(
            inputs=img_ctx,
            units=self._num_units,
            kernel_initializer=self.initializer,
            name='encoder_attention_image_ctx')
        inp = h0 + input_embedded + img_embedded
        hd = self.calc_hidden(inp)
        h1 = h0 + hd

        # masking
        h1 = (mask * h1) + ((1. - mask) * h0)
        print("H1: {}".format(h1))
        return h1, h1


def encoder_fn(img_ctx, cap, mask, temperature, params, mode, sen, slot_vocab):
    """

    :param img_ctx: (n, slots, c)
    :param cap:  (n, depth) [int]
    :param params:
    :return:
    """
    shape = tf.shape(img_ctx)
    n = shape[0]
    captions = tf.expand_dims(cap, axis=2)  # (n, depth, 1)
    caption_mask = tf.expand_dims(mask, axis=2)
    cell = EncoderStepCell(
        img_ctx=img_ctx,
        temperature=temperature,
        params=params,
        sen=sen,
        slot_vocab=slot_vocab,
        mode=mode,
        name='encoder_step_cell')
    initial_state = cell.build_hidden_state(n=n)
    output, states = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=(captions, caption_mask),
        initial_state=initial_state,
        time_major=False)
    print("new h1 {}".format(output))
    final_state = output[:, -1, :]  # (n, units)
    training = mode == tf.estimator.ModeKeys.TRAIN
    h = final_state
    for j in range(params.depth):
        h = tf.layers.dense(
            inputs=h,
            units=params.units,
            name='encoder_preds_{}'.format(j))
        h = leaky_relu(h)
        if params.encoder_dropout_hidden > 0:
            h = tf.layers.dropout(inputs=h, rate=params.encoder_dropout_hidden, training=training)
    raw_means = tf.layers.dense(
        inputs=h,
        units=params.vae_dim,
        name='encoder_vae_means')
    raw_std = tf.layers.dense(
        inputs=h,
        units=params.vae_dim,
        name='encoder_vae_stds')
    return raw_means, raw_std
