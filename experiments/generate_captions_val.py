import matplotlib

matplotlib.use('AGG')

import os

import tensorflow as tf

from visual_attention.generate_captions import generate_captions


def main(argv):
    model_dir = tf.flags.FLAGS.model_dir
    output_dir = os.path.join(model_dir, 'generated')
    generate_captions(model_dir=model_dir, output_dir=output_dir)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.flags.DEFINE_string('model-dir', 'output/model/img_ctx-nosen/v1', 'Model directory')
    tf.flags.DEFINE_string('model_dir', 'output/model/vocab/vae-imgsoft-v1', 'Model directory')
    tf.flags.DEFINE_string('batch-path', 'output/batches/val.npz', 'Batch path')
    tf.flags.DEFINE_string('cropped-path', 'output/cropped/val', 'Cropped path')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.flags.DEFINE_bool('deterministic', False, 'Deterministic')
    tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
    tf.app.run()
