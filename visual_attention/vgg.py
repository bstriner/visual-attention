import numpy as np
# from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import preprocess_input
# from tensorflow.contrib.keras.python.keras.applications.vgg19 import VGG19
from tensorflow.contrib.keras.api.keras.applications.vgg19 import VGG19


class VGG(object):
    def __init__(self):
        # self.input = tf.placeholder(tf.float32, [None,224,224,3], name='input_img')
        self.model = VGG19(include_top=False)

    def calc_features(self, img, sess):
        assert img.ndim == 4
        input = self.model.inputs[0]
        output = self.model.get_layer('block5_conv4').output
        x = (img.astype(np.float32) / 127.5) - 1.
        y = sess.run(output, feed_dict={input: x})
        return y
