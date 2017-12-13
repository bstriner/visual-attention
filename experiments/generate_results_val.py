import matplotlib

matplotlib.use('AGG')

import os

import tensorflow as tf

from visual_attention.generate_results import generate_results


def main(argv):
    model_dir = tf.flags.FLAGS.model_dir
    results_path = os.path.join(model_dir, 'results-val.json')
    generate_results(model_dir=model_dir, results_path=results_path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.flags.DEFINE_string('model_dir', 'output/model/vae/novae-512-soft', 'Model directory')
    #tf.flags.DEFINE_string('model_dir', 'output/model/vaecomparison/novaetest512v2', 'Model directory')
    tf.flags.DEFINE_string('model_dir', 'output/model/model512v4', 'Model directory')
    tf.flags.DEFINE_string('batch_path', 'output/batches/val.npz', 'Batch path')
    tf.flags.DEFINE_integer('batch_splits', 1, 'Batch splits')
    tf.flags.DEFINE_string('cropped_path', 'output/cropped/val', 'Cropped path')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.flags.DEFINE_bool('deterministic', True, 'Deterministic')
    tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
    tf.app.run()
