import matplotlib

matplotlib.use('AGG')

import os

import tensorflow as tf

from visual_attention.generate_results import generate_results


def main(argv):
    model_dir = tf.flags.FLAGS.model_dir
    results_path = os.path.join(model_dir, 'results-test.json')
    generate_results(model_dir=model_dir, results_path=results_path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model-dir', 'output/model/img_ctx-nosen/v1', 'Model directory')
    tf.flags.DEFINE_string('batch-path', 'output/batches/test-{}.npz', 'Batch path')
    tf.flags.DEFINE_integer('batch-splits', 10, 'Batch splits')
    tf.flags.DEFINE_string('cropped-path', 'output/cropped/test', 'Cropped path')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.flags.DEFINE_bool('deterministic', True, 'Deterministic')
    tf.app.run()
