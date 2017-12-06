import json
import os

import numpy as np
import tensorflow as tf

from visual_attention.make_experiment import experiment_fn


def main(_argv):
    model_dir = tf.flags.FLAGS.model_dir
    print("model_dir={}".format(model_dir))
    vocab = np.load('output/processed-annotations/vocab.npy')
    vocab_size = vocab.shape[0]
    run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)
    hparams = tf.contrib.training.HParams(lr=0.001,
                                          momentum=0.9,
                                          vocab_size=vocab_size,
                                          units=256,
                                          decay_rate=0.1,
                                          decay_steps=300000,
                                          smoothing=0.1,
                                          l2=1e-5,
                                          optimizer='adam',
                                          attn_mode_img='soft',
                                          batch_size=64)
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
    tf.flags.DEFINE_string('model-dir', 'output/model/v1',
                           'Model directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
