import json
import os

import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib.training import HParams


def get_hparams(model_dir):
    hparams = default_params()
    hparams_path = os.path.join(model_dir, 'configuration-hparams.json')
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparam_dict = json.load(f)
            for k, v in six.iteritems(hparam_dict):
                setattr(hparams, k, v)
    else:
        hparams.parse(tf.flags.FLAGS.hparams)
    return hparams


def default_params():
    vocab = np.load('output/processed-annotations/vocab.npy')
    vocab_size = vocab.shape[0]
    return HParams(
        lr=0.0001,
        depth=3,
        units=750,
        use_slot_vocab=False,
        momentum=0.9,
        frame_size=10,
        vocab_size=vocab_size,
        decay_rate=0.5,
        decay_steps=50000,
        vae_dim=100,
        kl_0=0.05,
        kl_anneal_steps=20000,
        smoothing=0.1,
        tau_0=1.,
        tau_decay_rate=0.80,
        tau_decay_steps=20000,
        tau_min=0.1,
        img_sen_l1=0.,
        img_sen_l2=1e-4,
        loss='nll',
        dropout_img_input=0.2,
        dropout_img_hidden=0.5,
        cap_dropout_input=0.2,
        cap_dropout_hidden=0.5,
        encoder_dropout_input=0.2,
        encoder_dropout_hidden=0.5,
        l2=1e-7,
        unity_reg=1e-3,
        optimizer='adam',
        attn_mode_img='soft',
        use_img_sen=False)
