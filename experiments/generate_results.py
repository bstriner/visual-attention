import matplotlib

matplotlib.use('AGG')

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator

from visual_attention.attention_model import model_fn
from visual_attention.feed_data import predict_input_fn, FeedFnHook
from visual_attention.util import token_id_to_vocab


def calc_caption(prediction, vocab):
    # Calculate generated caption
    caption = prediction['captions']
    assert caption.ndim == 1

    token_ids = []
    for c in caption:
        token_ids.append(c)
        if c == 0:
            break
    tokens = [token_id_to_vocab(i, vocab) for i in token_ids if i > 1]
    cap_string = ' '.join(tokens)
    return cap_string


def generate_results(model_dir, results_path):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    vocab = np.load('output/processed-annotations/vocab.npy')
    run_config = RunConfig(model_dir=model_dir)
    with open(os.path.join(model_dir, 'configuration-hparams.json')) as f:
        hparam_dict = json.load(f)
    hparams = HParams(**hparam_dict)
    print(hparams)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    val_path = tf.flags.FLAGS.batch_path
    hook = FeedFnHook(path_fmt=val_path, splits=1, batch_size=hparams.batch_size, predict=True)

    results = []
    for prediction in estimator.predict(input_fn=predict_input_fn, hooks=[hook]):
        caption = calc_caption(prediction=prediction, vocab=vocab)
        results.append({'image_id': prediction['image_id'], 'caption': caption})
    with open(results_path, 'w') as f:
        json.dump(results, f)


def main(argv):
    model_dir = tf.flags.FLAGS.model_dir
    results_path = os.path.join(model_dir, 'results.json')
    generate_results(model_dir=model_dir, results_path=results_path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model-dir', 'output/model/img_ctx/v2', 'Model directory')
    tf.flags.DEFINE_string('batch-path', 'output/batches/test.npy', 'Batch path')
    tf.flags.DEFINE_string('cropped-path', 'output/cropped/test', 'Cropped path')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.flags.DEFINE_bool('deterministic', True, 'Deterministic')
    tf.app.run()
