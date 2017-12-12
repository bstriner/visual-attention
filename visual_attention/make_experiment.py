import tensorflow as tf
from tensorflow.contrib.learn import Experiment
from tensorflow.python.estimator.estimator import Estimator

from .attention_model import model_fn
from .feed_data import input_fn, FeedFnHook


def experiment_fn(run_config, hparams):
    splits = 40
    train_path = 'output/batches/train-{}.npz'
    val_path = 'output/batches/val.npz'
    batch_size = tf.flags.FLAGS.batch_size
    train_hook = FeedFnHook(path_fmt=train_path, splits=splits, batch_size=batch_size)
    val_hook = FeedFnHook(path_fmt=val_path, splits=1, batch_size=batch_size)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    experiment = Experiment(
        estimator=estimator,
        train_input_fn=input_fn,
        eval_input_fn=input_fn,
        train_monitors=[train_hook],
        eval_hooks=[val_hook]
    )

    return experiment
