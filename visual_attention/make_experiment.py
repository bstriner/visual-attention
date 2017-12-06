import tensorflow as tf

from .attention_model import model_fn
from .feed_data import input_fn, FeedFnHook


def experiment_fn(run_config, hparams):
    splits = 20
    train_path = 'output/batches/train-{}.npz'
    val_path = 'output/batches/val.npz'
    train_hook = FeedFnHook(path_fmt=train_path, splits=splits)
    val_hook = FeedFnHook(path_fmt=val_path, splits=1)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=input_fn,
        eval_input_fn=input_fn,
        train_monitors=[train_hook],
        eval_hooks=[val_hook]
    )

    return experiment
