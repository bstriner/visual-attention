import tensorflow as tf

from visual_attention import training

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model_dir', 'output/model/vaecomparison/novaetest512v1', 'Model directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_string('hparams', 'vae_dim=0', 'Hyperparameters')
    tf.flags.DEFINE_bool('debug', False, 'Debug mode')
    tf.flags.DEFINE_bool('deterministic', False, 'Deterministic')
    tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
    tf.app.run(main=training.main)
