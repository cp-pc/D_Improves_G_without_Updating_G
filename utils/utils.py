import tensorflow as tf
import os


def init_sess():
    FLAGS = tf.app.flags.FLAGS
    #++ for specify GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    print(f"Using GPU: {FLAGS.gpu}")
    #++ =========================
    #++ For ignore log
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)
    #++ =========================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

