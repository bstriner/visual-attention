import json
import os

import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib.training import HParams


def get_hparams(model_dir, create):
    hparams = default_params()
    hparams_path = os.path.join(model_dir, 'configuration-hparams.json')
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparam_dict = json.load(f)
            for k, v in six.iteritems(hparam_dict):
                setattr(hparams, k, v)
    else:
        if create:
            hparams.parse(tf.flags.FLAGS.hparams)
            with open(hparams_path, 'w') as f:
                json.dump(hparams.values(), f)
        else:
            raise ValueError("No hparams file found: {}".format(hparams_path))
    return hparams


def default_params():
    vocab = np.load('output/processed-annotations/vocab.npy')
    vocab_size = vocab.shape[0]
    return HParams(
        lr=0.0001,
        depth=3,
        units=512,
        vae_dim=256,
        use_slot_vocab=True,
        use_img_sen=True,
        y0_feedback=False,
        attn_mode_img='soft',
        attn_mode_sen='gumbel',
        attn_mode_enc='gumbel',
        attn_mode_cap='gumbel',
        img_sen_l1=0.,
        img_sen_l2=0.,
        unity_reg=0.,
        momentum=0.9,
        frame_size=10,
        vocab_size=vocab_size,
        decay_rate=0.8,
        decay_steps=50000,
        kl_0=0.05,
        kl_anneal_steps=40000,
        smoothing=0.1,
        tau_0=1.,
        tau_decay_rate=0.70,
        tau_decay_steps=20000,
        tau_min=0.1,
        loss='nll',
        dropout_img_input=0.2,
        dropout_img_hidden=0.2,
        cap_dropout_input=0.2,
        cap_dropout_hidden=0.2,
        encoder_dropout_input=0.2,
        encoder_dropout_hidden=0.2,
        l2=1e-7,
        optimizer='adam')
