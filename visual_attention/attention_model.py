import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, apply_regularization

from .caption_model import train_decoder_fn, predict_decoder_fn
from .encoder_model import encoder_fn
from .gumbel import get_temperature
from .image_model import slot_vocab_fn, attention_fn, apply_attn
from .losses import nll_loss, cross_entropy_loss
from .util import EPSILON, get_kl_weight


def model_fn(features, labels, mode, params):
    img = features['images']  # (image_n,h, w, c)

    temperature = get_temperature(params)
    img_attn, img_sen = attention_fn(img, temperature=temperature, mode=mode, params=params)

    if params.use_slot_vocab:
        img_ctx = apply_attn(img=img, att=img_attn)  # , sen=img_sen)  # (image_n, frames, c)
        slot_vocab = slot_vocab_fn(img_ctx=img_ctx, params=params)  # (image_n, frames, vocab+1)
        if img_sen is not None:
            slot_vocab *= tf.expand_dims(img_sen, axis=2)
    else:
        img_ctx = apply_attn(img=img, att=img_attn, sen=img_sen)  # (image_n, frames, c)
        slot_vocab = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        n = tf.shape(img_ctx)[0]
        if params.vae_dim > 0:
            enc = tf.random_normal(shape=(n, params.vae_dim), mean=0, stddev=1, dtype=tf.float32)
        else:
            enc = None
        logits, slot_attn, slot_sentinel, y1 = predict_decoder_fn(
            slot_vocab=slot_vocab,
            img_ctx=img_ctx,
            sen=img_sen,
            enc=enc,
            params=params,
            depth=30,
            temperature=temperature,
            mode=mode)
        predictions = {
            'captions': y1,
            'image_ids': tf.get_default_graph().get_tensor_by_name('image_ids:0'),
            'slot_attention': slot_attn,
            'slot_sentinel': slot_sentinel,
            'image_attention': img_attn
        }
        if slot_vocab is not None:
            predictions['slot_vocab'] = slot_vocab
        if img_sen is not None:
            predictions['image_sentinel'] = img_sen

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        raw_cap = features['captions']  # (caption_n, depth)
        cap = tf.maximum(raw_cap - 1, 0)  # (caption_n, depth)
        cap_mask = 1. - tf.cast(tf.equal(raw_cap, 0), tf.float32)  # (caption_n, depth)
        ass = features['assignments']  # (caption_n,)
        if slot_vocab is not None:
            decoder_vocab = tf.gather(slot_vocab, ass, axis=0)  # (caption_n, frames, c)
        else:
            decoder_vocab = None
        if img_sen is not None:
            decoder_sen = tf.gather(img_sen, ass, axis=0)  # (caption_n, frames)
        else:
            decoder_sen = None
        decoder_img_ctx = tf.gather(img_ctx, ass, axis=0)
        if params.vae_dim > 0:
            mu, raw_sig = encoder_fn(img_ctx=decoder_img_ctx,
                                     sen=decoder_sen, slot_vocab=decoder_vocab,
                                     mask=cap_mask,
                                     cap=cap, temperature=temperature, params=params, mode=mode)
            sigma = EPSILON + tf.exp(raw_sig)
            enc = mu + (sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32))
            kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(EPSILON + tf.square(sigma)) - 1, 1)
            kl_loss = tf.reduce_mean(kl_loss, 0)
            tf.summary.scalar('kl_loss', kl_loss)
        else:
            enc = None

        logits, slot_attn, slot_sentinel = train_decoder_fn(
            slot_vocab=decoder_vocab,
            img_ctx=decoder_img_ctx,
            sen=decoder_sen,
            cap=cap,
            temperature=temperature,
            params=params,
            mode=mode,
            enc=enc)

        # Loss
        if params.loss == 'cross_entropy':
            loss = tf.reduce_mean(cross_entropy_loss(
                labels=cap,
                mask=cap_mask,
                logits=logits,
                smoothing=params.smoothing))
        elif params.loss == 'nll':
            loss = tf.reduce_mean(nll_loss(
                labels=cap,
                mask=cap_mask,
                logits=logits,
                mean=False))
        else:
            raise ValueError()

        if params.vae_dim > 0:
            kl_weight = get_kl_weight(params)
            loss += kl_weight * kl_loss

        # Regularization
        # slot_attn: (n, depth, frame_size)
        # slot_sentinel: (n, depth, 1)
        if params.l2 > 0:
            reg = apply_regularization(l2_regularizer(params.l2), tf.trainable_variables())
            tf.summary.scalar("regularization", reg)
            loss += reg
        if params.unity_reg > 0 and decoder_sen is not None:
            slot_sum = tf.reduce_sum(tf.expand_dims(cap_mask, 2) * slot_attn * slot_sentinel, axis=1)  # (n, frame_size)
            slot_diff = tf.square(slot_sum - decoder_sen)
            unity_regularization = params.unity_reg * tf.reduce_mean(tf.reduce_sum(slot_diff, 1))
            tf.summary.scalar("unity_regularization", unity_regularization)
            loss += unity_regularization
        if (params.img_sen_l1 > 0 or params.img_sen_l2 > 0) and img_sen is not None:
            img_sen_reg = 0
            if params.img_sen_l1 > 0:
                img_sen_reg += params.img_sen_l1 * tf.reduce_mean(tf.reduce_sum(img_sen, axis=1), axis=0)
            if params.img_sen_l2 > 0:
                img_sen_reg += params.img_sen_l2 * tf.reduce_mean(tf.square(tf.reduce_sum(img_sen, axis=1)), axis=0)
            tf.summary.scalar('image_sentinel_regularization', img_sen_reg)
            loss += img_sen_reg

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
