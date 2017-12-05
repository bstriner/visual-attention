import numpy as np

from visual_attention.image_processing import read_image, write_image, crop_image
from visual_attention.vgg import VGG
import tensorflow as tf

def main():
    path = 'testimage.jpg'
    x = read_image(path)
    print(type(x))
    print(x.shape)
    c = crop_image(x)
    write_image('cropped.jpg', c)


    with tf.Session() as sess:
        m = VGG()
        batch = np.expand_dims(c, axis=0)
        ret = m.calc_features(batch, sess)
        print(ret.shape)
        print(ret[0,0][:50])


if __name__ == '__main__':
    main()
