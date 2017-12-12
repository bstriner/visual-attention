import json
import os

import tensorflow as tf
from tensorflow.contrib.learn import RunConfig

from visual_attention.debugging import enable_debugging_monkey_patch
from visual_attention.default_params import get_hparams
from visual_attention.make_experiment import experiment_fn


def main(_argv):
    if tf.flags.FLAGS.debug:
        enable_debugging_monkey_patch()
    model_dir = tf.flags.FLAGS.model_dir
    print("model_dir={}".format(model_dir))
    # vocab = np.load('output/processed-annotations/vocab.npy')
    # vocab_size = vocab.shape[0]
    run_config = RunConfig(model_dir=model_dir)
    hparams = get_hparams(model_dir)
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
