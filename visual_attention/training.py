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
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir={}".format(model_dir))
    run_config = RunConfig(model_dir=model_dir)
    hparams = get_hparams(model_dir, create=True)
    estimator = tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)
