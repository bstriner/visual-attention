import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, apply_regularization

from .caption_model import PredictStepCell, TrainStepCell
from .gumbel import gumbel_softmax, gumbel_sigmoid, softmax_nd

EPSILON = 1e-7


def leaky_relu(x):
    return tf.maximum(x, x * 0.2)
