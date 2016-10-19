import tensorflow as tf
from tensorflow.contrib import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from prepare_data import my_shuffle


def create_inputs():

    batch_size = 2

    sess = tf.Session()
    f = []
    for (dirpath, dirnames, filenames) in os.walk('training_set'):
        f.extend(map(lambda x: 'training_set/' + x, filenames))
        break
    print f[0]
    files = tf.train.string_input_producer(f)
    reader = tf.FixedLengthRecordReader(record_bytes = 3054)
    _, value = reader.read(files)

    record_bytes = tf.decode_raw(value, tf.uint8)
    record_bytes = tf.reshape(record_bytes, [-1, 1])

    lr_image = tf.reshape(record_bytes[0: 17 * 17 * 3], (17, 17, 3))
    hr_data = tf.reshape(record_bytes[17 * 17 * 3:], (9, 9, 27))

    return lr_image, hr_data

create_inputs()
