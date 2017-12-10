import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

from .gumbel import gumbel_softmax, gumbel_sigmoid, sample_one_hot, sample_argmax, sample_sigmoid

EPSILON = 1e-7


class BaseStepCell(RNNCell):
    def __init__(
            self,
            sen,
            temperature,
            slot_vocab,
            img_ctx,
            params,
            mode,
            activation=tf.nn.relu,
            reuse=None,
            name=None):
        super(BaseStepCell, self).__init__(_reuse=reuse, name=name)
        self.activation = activation
        self.sen = sen
        self._num_units = params.units
        self.vocab_size = params.vocab_size
        self.frame_size = params.frame_size
        self.dropout_input = params.cap_dropout_input
        self.dropout_hidden = params.cap_dropout_hidden
        self.temperature = temperature
        self.slot_vocab = slot_vocab
        self.img_ctx=img_ctx
        self.mode = mode
        self.training = mode == tf.estimator.ModeKeys.TRAIN
        scale = 0.05
        self.initializer = tf.initializers.random_uniform(minval=-scale, maxval=scale)

    @property
    def state_size(self):
        return self._num_units

    def build_hidden_state(self, n):
        h0 =  tf.get_variable(
            name='caption_h0',
            shape=[1, self._num_units],
            trainable=True,
            initializer=self.initializer)
        h0_tiled = tf.tile(h0,[n,1])
        return h0_tiled

    @property
    def output_size(self):
        return self.vocab_size + 2

    def calc_step_output(self, inp):
        # Hidden representation
        h = inp
        if self.dropout_input > 0:
            h = tf.layers.dropout(inputs=h, rate=self.dropout_input, training=self.training)
        for j in range(3):
            h = tf.layers.dense(
                inputs=h,
                units=self._num_units,
                kernel_initializer=self.initializer,
                name='input_attn{}'.format(j))
            h = self.activation(h)
            if self.dropout_hidden > 0:
                h = tf.layers.dropout(inputs=h, rate=self.dropout_hidden, training=self.training)
        generate_hidden = h
        # Attention over slots
        attn_logits = tf.layers.dense(
            inputs=generate_hidden,
            units=self.frame_size,
            name='input_attn_final')
        #attn_bias = tf.log(1e-7 + self.sen)
        #attn_logits += attn_bias
        with tf.name_scope('recurrent_slot_attention'):
            if self.mode == tf.estimator.ModeKeys.PREDICT:
                if tf.flags.FLAGS.deterministic:
                    slot_attn = tf.one_hot(tf.argmax(attn_logits, axis=-1),
                                           depth=self.frame_size, axis=-1)
                else:
                    slot_attn = sample_one_hot(logits=attn_logits, axis=-1)
            else:
                slot_attn = gumbel_softmax(logits=attn_logits, temperature=self.temperature, axis=-1)

        # Gate between slots and generation
        sent_logits = tf.layers.dense(
            inputs=generate_hidden,
            units=1,
            name='input_sent',
            kernel_initializer=self.initializer)
        with tf.name_scope('recurrent_slot_sentinel'):
            if self.mode == tf.estimator.ModeKeys.PREDICT:
                if tf.flags.FLAGS.deterministic:
                    slot_sentinel = tf.cast(tf.greater(sent_logits, 0), tf.float32)
                else:
                    slot_sentinel = sample_sigmoid(logits=sent_logits)
            else:
                slot_sentinel = gumbel_sigmoid(logits=sent_logits, temperature=self.temperature)
        # end token
        end_token_logits = tf.layers.dense(
            inputs=generate_hidden,
            units=1,
            name='end_token')
        if self.slot_vocab is not None:
            # Generate next vocab token
            next_token_generated_logits = tf.layers.dense(
                inputs=generate_hidden,
                units=self.vocab_size + 1,  # [unknown]+vocab
                name='next_token',
                kernel_initializer=self.initializer)
            # Pull Vocab token
            with tf.control_dependencies([
                tf.assert_rank(slot_attn, 2),
                tf.assert_rank(self.slot_vocab, 3)]):
                # Next vocab given slots
                next_token_slot_logits = tf.reduce_sum(self.slot_vocab * tf.expand_dims(slot_attn, 2), axis=1)
                # (n, vocab+1)
            # Mix generation and selection
            next_token_logits = ((slot_sentinel * next_token_slot_logits) +
                                 ((1. - slot_sentinel) * next_token_generated_logits))
        else:
            img_ctx_attn_raw = tf.reduce_sum(self.img_ctx * tf.expand_dims(slot_attn, 2), axis=1)
            img_ctx_attn = tf.layers.dense(
                inputs=img_ctx_attn_raw,
                units=self._num_units,
                name='img_ctx_attn',
                kernel_initializer=self.initializer)
            h = img_ctx_attn + generate_hidden
            #if self.dropout_input > 0:
            #    h = tf.layers.dropout(inputs=h, rate=self.dropout_input, training=self.training)
            for i in range(3):
                h = tf.layers.dense(
                inputs=h,
                units=self._num_units,
                name='img_ctx_attn_{}'.format(i),
                kernel_initializer=self.initializer)
                h = self.activation(h)
                if self.dropout_hidden > 0:
                    h = tf.layers.dropout(inputs=h, rate=self.dropout_hidden, training=self.training)
            next_token_logits = tf.layers.dense(
                inputs=h,
                units=self.vocab_size+1,
                name='img_ctx_attn_token',
                kernel_initializer=self.initializer)
        # Combine with end token
        output_logits = tf.concat([end_token_logits, next_token_logits - end_token_logits], axis=1)
        # (n, vocab+2) [end, unk]+vocab
        return output_logits, slot_attn, slot_sentinel

    def calc_step_hidden(self, inp, y0, slot_attn, slot_sentinel):
        # embed y0
        y_embed = tf.get_variable(
            name='y_embed',
            shape=[self.vocab_size + 2, self._num_units],  # [end, unknown] + vocab
            trainable=True,
            initializer=self.initializer)
        y_embedded = tf.gather(y_embed, y0, axis=0)  # n, unit
        # embed attention
        slot_combined_attn = slot_sentinel * slot_attn  # (n, frame_size)
        slot_ctx0 = tf.layers.dense(
            inputs=slot_combined_attn,
            units=self._num_units,
            kernel_initializer=self.initializer,
            name='slot_ctx0')
        slot_ctx1 = tf.layers.dense(
            inputs=1. - slot_combined_attn,
            units=self._num_units,
            kernel_initializer=self.initializer,
            name='slot_ctx1')
        slot_ctx = slot_ctx0 + slot_ctx1
        # Calculate hidden state
        h = inp + y_embedded + slot_ctx
        if self.dropout_input > 0:
            h = tf.layers.dropout(inputs=h, rate=self.dropout_input, training=self.training)
        for j in range(3):
            h = tf.layers.dense(
                inputs=h,
                units=self._num_units,
                kernel_initializer=self.initializer,
                name='forward{}'.format(j))
            h = self.activation(h)
            if self.dropout_hidden > 0:
                h = tf.layers.dropout(inputs=h, rate=self.dropout_hidden, training=self.training)
        hd = tf.layers.dense(
            inputs=h,
            units=self._num_units,
            kernel_initializer=self.initializer,
            name='forwardfinal')
        return hd

    def calc_sen_ctx(self):
        sen_ctx0 = tf.layers.dense(
            inputs=self.sen,
            units=self._num_units,
            kernel_initializer=self.initializer,
            name='sen_ctx0')
        sen_ctx1 = tf.layers.dense(
            inputs=1. - self.sen,
            units=self._num_units,
            kernel_initializer=self.initializer,
            name='sen_ctx1')
        return sen_ctx0 + sen_ctx1


class TrainStepCell(BaseStepCell):
    def __init__(self, sen, temperature, slot_vocab, img_ctx, params, mode, reuse=None, name=None):
        super(TrainStepCell, self).__init__(
            sen=sen,
            temperature=temperature,
            mode=mode,
            name=name,
            slot_vocab=slot_vocab,
            img_ctx=img_ctx,
            params=params,
            reuse=reuse)

    def call(self, inputs, state):
        y0 = tf.squeeze(inputs, axis=1)  # (n,) [end, unknown]+vocab
        h0 = state
        sen_ctx = self.calc_sen_ctx()
        inp = sen_ctx + h0
        output_logits, slot_attn, slot_sentinel = self.calc_step_output(inp=inp)
        h1 = self.calc_step_hidden(
            inp=inp,
            y0=y0,
            slot_attn=slot_attn,
            slot_sentinel=slot_sentinel)
        return (output_logits, slot_attn, slot_sentinel), h1

    @property
    def output_size(self):
        return self.vocab_size + 2, self.frame_size, 1


class PredictStepCell(BaseStepCell):
    def __init__(self, sen, slot_vocab, img_ctx, params, mode, reuse=None, name=None):
        super(PredictStepCell, self).__init__(
            sen=sen,
            temperature=None,
            name=name,
            slot_vocab=slot_vocab,
            img_ctx=img_ctx,
            params=params,
            reuse=reuse,
            mode=mode)

    def call(self, inputs, state):
        h0 = state
        sen_ctx = self.calc_sen_ctx()
        inp = sen_ctx + h0
        output_logits, slot_attn, slot_sentinel = self.calc_step_output(inp=inp)
        # y0 = sample_argmax(output_logits, axis=-1)  # [end, unk] + vocab
        # y0 = tf.argmax(output_logits, axis=-1)
        if tf.flags.FLAGS.deterministic:
            y0 = tf.argmax(output_logits, axis=-1)
        else:
            y0 = sample_argmax(output_logits, axis=-1)
        h1 = self.calc_step_hidden(
            inp=inp,
            y0=y0,  # [end, unk] + vocab
            slot_attn=slot_attn,
            slot_sentinel=slot_sentinel)
        yout = tf.cast(tf.expand_dims(y0, 1), tf.float32)
        return (output_logits, slot_attn, slot_sentinel, yout), h1

    @property
    def output_size(self):
        return self.vocab_size + 2, self.frame_size, 1, 1


def shift_captions(cap):
    n = tf.shape(cap)[0]
    zeros = tf.zeros((n, 1), dtype=tf.int32)
    shifted = tf.concat((zeros, cap[:, :-1] + 1), axis=1)
    return shifted


def train_decoder_fn(slot_vocab, sen, cap, img_ctx, temperature, params, mode):
    """

    :param slot_vocab:
    :param sen:
    :param cap:
    :param temperature:
    :param params:
    :param mode:
    :return:
    """
    shape = tf.shape(img_ctx)
    n = shape[0]
    captions = tf.expand_dims(cap, axis=2)  # (n, depth, 1)
    cell = TrainStepCell(
        sen=sen,
        img_ctx=img_ctx,
        temperature=temperature,
        slot_vocab=slot_vocab,
        params=params,
        mode=mode,
        name='step_cell')
    #initial_state = cell.zero_state(batch_size=n, dtype=tf.float32)
    initial_state = cell.build_hidden_state(n=n)
    (hlogits, slot_attn, slot_sentinel), h1 = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=captions,
        initial_state=initial_state,
        time_major=False)
    return hlogits, slot_attn, slot_sentinel  # (n, depth, vocab)


def predict_decoder_fn(slot_vocab, img_ctx, sen, params, mode, depth):
    shape = tf.shape(img_ctx)
    n = shape[0]
    cell = PredictStepCell(
        sen=sen,
        slot_vocab=slot_vocab,
        params=params,
        img_ctx=img_ctx,
        mode=mode,
        name='step_cell')
    #initial_state = cell.zero_state(batch_size=n, dtype=tf.float32)
    initial_state = cell.build_hidden_state(n=n)
    inputs = tf.zeros((1, depth, 1))
    (logits, slot_attn, slot_sentinel, y1), state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=inputs,
        initial_state=initial_state,
        time_major=False)
    return logits, slot_attn, slot_sentinel, tf.cast(tf.squeeze(y1, 2), tf.int32)
