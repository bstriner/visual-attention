import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams

from visual_attention.debugging import enable_debugging_monkey_patch
from visual_attention.make_experiment import experiment_fn


def main(_argv):
    batch_size = 64
    if tf.flags.FLAGS.debug:
        enable_debugging_monkey_patch()
        batch_size = 1
    model_dir = tf.flags.FLAGS.model_dir
    print("model_dir={}".format(model_dir))
    vocab = np.load('output/processed-annotations/vocab.npy')
    vocab_size = vocab.shape[0]
    run_config = RunConfig(model_dir=model_dir)
    hparams = HParams(lr=0.0001,
                      use_slot_vocab=False,
                      momentum=0.9,
                      frame_size=10,
                      vocab_size=vocab_size,
                      units=750,
                      decay_rate=0.5,
                      decay_steps=300000,
                      smoothing=0.1,
                      tau_0=1.,
                      tau_decay_rate=0.50,
                      tau_decay_steps=20000,
                      tau_min=0.1,
                      img_sen_l1=0.,
                      img_sen_l2=1e-4,
                      loss='nll',
                      dropout_img_input=0.2,
                      dropout_img_hidden=0.5,
                      cap_dropout_input=0.2,
                      cap_dropout_hidden=0.5,
                      l2=1e-7,
                      unity_reg=1e-2,
                      optimizer='adam',
                      attn_mode_img='soft',
                      batch_size=batch_size)
    hparams.parse(tf.flags.FLAGS.hparams)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'configuration-flags.json'), 'w') as f:
        json.dump(tf.flags.FLAGS.__flags, f)
    with open(os.path.join(model_dir, 'configuration-hparams.json'), 'w') as f:
        json.dump(hparams.values(), f)

    estimator = tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model-dir', 'output/model/img_ctx/v3', 'Model directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.flags.DEFINE_bool('deterministic', False, 'Deterministic')
    tf.app.run()
