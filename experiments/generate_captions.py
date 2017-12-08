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


def token_id_to_vocab(token_id, vocab):
    token_id = int(np.asscalar(token_id))
    # print("Token: {}, {}".format(token_id, type(token_id)))
    if token_id == 0:
        return '_END_'
    elif token_id == 1:
        return '_UNK_'
    else:
        v = vocab[token_id - 2].decode('ascii')
        # print("v: {},{}".format(v, type(v)))
        return v


def write_prediction(output_path, prediction, vocab):
    # print(type(prediction))
    # print(prediction.keys())

    images = []

    # Read cropped image
    id = np.asscalar(prediction['image_ids'])
    img_path = 'output/cropped/val/{:012d}.jpg'.format(id)
    assert os.path.exists(img_path)
    img = cv2.imread(img_path)
    # print("Image: {}".format(type(img)))
    images.append((img, 'original'))

    # Image attention maps
    image_sentinel = prediction['image_sentinel']
    image_attention = prediction['image_attention']
    assert image_sentinel.ndim == 1
    assert image_attention.ndim == 3
    for i, s in enumerate(image_sentinel):
        if s > 0.5:
            attn_img = image_attention[:, :, i]
            attn_img = cv2.resize(attn_img, (224, 224))
            images.append((attn_img, 'slot {}'.format(i)))

    # Write images
    n = len(images)
    f, axs = plt.subplots(nrows=1, ncols=n)
    for i, (im, name) in enumerate(images):
        # print(type(im))
        # print(im.shape)
        axs[i].imshow(im)
        axs[i].set_title(name)
        axs[i].axis('off')
    f.savefig(output_path + '.png')
    plt.close(f)

    # Calculate generated caption
    caption = prediction['captions']
    slot_sentinel = prediction['slot_sentinel']
    slot_attention = prediction['slot_attention']
    caption = np.squeeze(caption, 1)
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
    val_path = 'output/batches/val.npz'
    hook = FeedFnHook(path_fmt=val_path, splits=1, batch_size=hparams.batch_size, predict=True)
    for i, prediction in enumerate(estimator.predict(input_fn=predict_input_fn, hooks=[hook])):
        write_prediction(os.path.join(output_dir, '{:08d}'.format(i)),
                         prediction=prediction, vocab=vocab)
        if i > 100:
            break


def main(argv):
    output_dir = 'output/generated'
    model_dir = tf.flags.FLAGS.model_dir
    generate_captions(model_dir=model_dir, output_dir=output_dir)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model-dir', 'output/model/v01',
                           'Model directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.app.run()
