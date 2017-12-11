import matplotlib

matplotlib.use('AGG')

import csv
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator

from visual_attention.attention_model import model_fn
from visual_attention.feed_data import predict_input_fn, FeedFnHook
from visual_attention.util import token_id_to_vocab


def write_prediction(output_path, prediction, vocab, use_slot_vocab):
    # print(type(prediction))
    # print(prediction.keys())

    images = []

    # Read cropped image
    id = np.asscalar(prediction['image_ids'])
    img_path = os.path.join(tf.flags.FLAGS.cropped_path, '{:012d}.jpg'.format(id))
    assert os.path.exists(img_path)
    img = cv2.imread(img_path)
    # print("Image: {}".format(type(img)))
    images.append((img, 'original'))

    if use_slot_vocab:
        slot_vocab = prediction['slot_vocab']  # (slots, vocab+1)
        assert slot_vocab.ndim == 2
        slot_tokens = [token_id_to_vocab(i + 1, vocab=vocab) for i in np.argmax(slot_vocab, axis=-1)]

    # Image attention maps
    image_attention = prediction['image_attention']
    assert image_attention.ndim == 3
    if 'image_sentinel' in prediction:
        image_sentinel = prediction['image_sentinel']
    else:
        image_sentinel = np.ones((image_attention.shape[2],))
    assert image_sentinel.ndim == 1
    for i, s in enumerate(image_sentinel):
        if s > 0.5:
            attn_img = image_attention[:, :, i]
            attn_img = cv2.resize(attn_img, (224, 224))
            if use_slot_vocab:
                images.append((attn_img, '{}({})'.format(slot_tokens[i], i)))
            else:
                images.append((attn_img, 'slot {}'.format(i)))

    # Write images
    n = len(images)
    figsize = 2
    if n > 1:
        f, axs = plt.subplots(nrows=1, ncols=n, figsize=(n * figsize, figsize))
        for i, (im, name) in enumerate(images):
            # print(type(im))
            # print(im.shape)
            ax = axs[i]
            ax.imshow(im)
            ax.set_title(name)
            ax.axis('off')
    else:
        f = plt.figure(figsize=(figsize, figsize))
        plt.imshow(images[0][0])
        plt.title(images[0][1])
        plt.axis('off')
    f.savefig(output_path + '.png')
    plt.close(f)

    # Calculate generated caption
    caption = prediction['captions']
    slot_sentinel = prediction['slot_sentinel']
    slot_attention = prediction['slot_attention']
    slot_sentinel = np.squeeze(slot_sentinel, 1)
    assert caption.ndim == 1
    assert slot_sentinel.ndim == 1
    assert slot_attention.ndim == 2

    token_ids = []
    for c in caption:
        token_ids.append(c)
        if c == 0:
            break
    tokens = [token_id_to_vocab(i, vocab) for i in token_ids]

    refs = []
    for sen, att in zip(slot_sentinel, slot_attention):
        if sen < 0.5:
            refs.append('')
        else:
            refs.append('({})'.format(np.asscalar(np.argmax(att))))
    refs = refs[:len(tokens)]
    # print(tokens)
    # print(refs)

    cap_strings = [t + r for t, r in zip(tokens, refs)]
    cap_string = ' '.join(cap_strings)
    with open(output_path + '.txt', 'w') as f:
        f.write(cap_string)
        f.write("\n")

    return cap_string
    # raise ValueError()


def generate_captions(model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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
    use_slot_vocab = hparams.use_slot_vocab
    hook = FeedFnHook(path_fmt=val_path, splits=1, batch_size=hparams.batch_size, predict=True)
    with open(os.path.join(output_dir, 'captions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Index', 'Caption'])
        for i, prediction in enumerate(estimator.predict(input_fn=predict_input_fn, hooks=[hook])):
            caption = write_prediction(os.path.join(output_dir, '{:08d}'.format(i)),
                                       prediction=prediction, vocab=vocab, use_slot_vocab=use_slot_vocab)
            w.writerow([i, caption])
            if i > 100:
                break


def main(argv):
    model_dir = tf.flags.FLAGS.model_dir
    output_dir = os.path.join(model_dir, 'generated')
    generate_captions(model_dir=model_dir, output_dir=output_dir)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model-dir', 'output/model/img_ctx-nosen/v1', 'Model directory')
    tf.flags.DEFINE_string('batch-path', 'output/batches/val.npz', 'Batch path')
    tf.flags.DEFINE_string('cropped-path', 'output/cropped/val', 'Cropped path')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.flags.DEFINE_bool('deterministic', False, 'Deterministic')
    tf.app.run()
