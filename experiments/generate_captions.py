import json
import os

from tensorflow.contrib.training import HParams
import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from visual_attention.attention_model import model_fn
from tensorflow.contrib.learn import RunConfig
from visual_attention.feed_data import predict_input_fn, FeedFnHook

def main(argv):
    output_dir = 'output/generated'
    model_dir = tf.flags.FLAGS.model_dir
    run_config = RunConfig(model_dir=model_dir)
    with open(os.path.join(model_dir, 'configuration-hparams.json')) as f:
        hparam_dict = json.load(f)
    hparams = HParams(**hparam_dict)
    print(hparams)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    val_path = 'output/batches/val.npz'
    hook = FeedFnHook(path_fmt=val_path, splits=1, batch_size=hparams.batch_size, predict=True)
    for prediction in estimator.predict(input_fn=predict_input_fn, hooks=[hook]):
        print(type(prediction))
        print(prediction.keys())
        raise ValueError()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model-dir', 'output/model/v05',
                           'Model directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.app.run()