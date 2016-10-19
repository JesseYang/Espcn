from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import json
import time

import tensorflow as tf
from reader import create_inputs
from espcn_net import EspcnNet

BATCH_SIZE = 32
NUM_STEPS = 100000
LEARNING_RATE = 0.0001
LOGDIR_ROOT = './logdir'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

def get_arguments():
    parser = argparse.ArgumentParser(description='EspcnNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many image files to process at once.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--logdir_root', type=str, default=LOGDIR_ROOT,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')

def get_default_logdir(logdir_root):
    print(logdir_root)
    print(STARTED_DATESTRING)
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir

def check_params(args, params):
    if len(params['filters_size']) - len(params['channels']) != 1:
        print("The length of 'filters_size' must be greater then the length of 'channels' by 1.")
        return False
    return True

def main():
    args = get_arguments()

    with open("./params.json", 'r') as f:
        params = json.load(f)

    if check_params(args, params) == False:
        return

    logdir_root = args.logdir_root
    logdir = get_default_logdir(logdir_root)


    lr_image, hr_data = create_inputs()

    queue = tf.FIFOQueue(
        256,
        ['uint8', 'uint8'],
        shapes=[(params['lr_size'], params['lr_size'], 3),
                (params['lr_size'] - params['edge'], params['lr_size'] - params['edge'], 3 * params['ratio']**2)])
    enqueue = queue.enqueue([lr_image, hr_data])
    batch_input = queue.dequeue_many(args.batch_size)

    net = EspcnNet(filters_size=params['filters_size'],
                   channels=params['channels'],
                   ratio=params['ratio'])
    loss = net.loss(batch_input)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # set up logging for tensorboard
    writer = tf.train.SummaryWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    summaries = tf.merge_all_summaries()

    # set up session
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    coord = tf.train.Coordinator()
    qr = tf.train.QueueRunner(queue, [enqueue])
    qr.create_threads(sess, coord=coord, start=True)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # saver for storing checkpoints of the model
    saver = tf.train.Saver()


    try:
        print(args.num_steps)
        start_time = time.time()
        for step in range(args.num_steps):
            summary, loss_value, _ = sess.run([summaries, loss, optim])
            loss_value, _ = sess.run([loss, optim])
            writer.add_summary(summary, step)

            if step % 100 == 0 and step > 0:
                duration = time.time() - start_time
                print('step {:d} - loss = {:.9f}, ({:.3f} sec/100 step)'.format(step, loss_value, duration))
                start_time = time.time()
                save(saver, sess, logdir, step)
                last_saved_step = step
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        # if step > last_saved_step:
            # save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
