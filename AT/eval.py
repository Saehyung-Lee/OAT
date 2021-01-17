"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import sys

import numpy as np
import tensorflow as tf

import cifar10_input
from model import Model
from pgd_attack import LinfPGDAttack

num_classes = 10
num_steps = int(sys.argv[1])
loss_func = sys.argv[2]
model_dir = sys.argv[3]

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
data_path = config['data_path']
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
model = Model(num_classes=num_classes)

scaler_eval = 1.0

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       num_steps,
                       config['step_size'],
                       config['random_start'],
                       loss_func)

saver = tf.train.Saver(max_to_keep=5)

import tqdm

def evaluate(sess):
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_corr_adv = 0
    for ibatch in tqdm.tqdm(range(num_batches)):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = raw_cifar.eval_data.xs[bstart:bend, :]
        y_batch = raw_cifar.eval_data.ys[bstart:bend]
        y_batch = np.eye(num_classes)[y_batch]  # one hot coding

        x_batch_adv = attack.perturb(x_batch, y_batch, sess, scaler_eval, is_training=False)

        dict_adv = {model.x_input: x_batch_adv,
                    model.is_training: False,
                    model.scaler: scaler_eval,
                    model.y_input: y_batch
                    }

        cur_corr_adv = sess.run(model.num_correct, feed_dict=dict_adv)

        total_corr_adv += cur_corr_adv

    acc_adv = total_corr_adv / num_eval_examples
    print(sys.argv[1] + ' ' + sys.argv[2] + ' ' + sys.argv[3])
    print(f'acc_adv: {acc_adv}')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # restore
    cur_checkpoint = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, cur_checkpoint)
    # evaluate
    evaluate(sess)
