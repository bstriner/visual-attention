import glob
import os

import numpy as np
import tensorflow as tf

from .image_processing import read_image, write_image, crop_image
from .vgg import VGG


class Preprocessor(object):
    def __init__(self):
        self.model = VGG()

    def preprocess_image(self, path, crop_out, feature_out, sess):
        im = read_image(path)
        cropped = crop_image(im)
        write_image(crop_out, cropped)
        batch = np.expand_dims(cropped, axis=0)
        ret = self.model.calc_features(batch, sess)
        y = np.squeeze(ret, axis=0)
        np.save(feature_out, y)


def preprocess_dir(image_path, crop_path, feature_path):
    os.makedirs(crop_path, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)
    with tf.Session() as sess:
        proc = Preprocessor()
        for file in glob.glob(r'{}/*.jpg'.format(image_path)):
            basename = os.path.splitext(os.path.basename(file))[0]
            crop_out = os.path.join(crop_path, basename + '.jpg')
            feature_out = os.path.join(feature_path, basename + '.npy')
            proc.preprocess_image(file, crop_out, feature_out, sess=sess)
