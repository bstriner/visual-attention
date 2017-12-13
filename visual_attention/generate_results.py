import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.python.estimator.estimator import Estimator
from tqdm import tqdm

from visual_attention.attention_model import model_fn
from visual_attention.feed_data import predict_input_fn, FeedFnHook
from visual_attention.util import token_id_to_vocab
from .default_params import get_hparams


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
    hparams = get_hparams(model_dir=model_dir, create=False)
    print(hparams)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    val_path = tf.flags.FLAGS.batch_path
    splits = tf.flags.FLAGS.batch_splits
    batch_size = tf.flags.FLAGS.batch_size
    hook = FeedFnHook(path_fmt=val_path, splits=splits, batch_size=batch_size, predict=True, single_pass=True)

    results = []
    it = tqdm(desc='Generating results')
    for prediction in estimator.predict(input_fn=predict_input_fn, hooks=[hook]):
        caption = calc_caption(prediction=prediction, vocab=vocab)
        results.append({'image_id': np.asscalar(prediction['image_ids']), 'caption': caption})
        it.update(1)
    with open(results_path, 'w') as f:
        json.dump(results, f)
