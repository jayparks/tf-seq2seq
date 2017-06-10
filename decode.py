
#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
import json
import random

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from data.data_iterator import TextIterator

import data.util as util
import data.data_utils as data_utils
from data.data_utils import prepare_batch
from data.data_utils import prepare_train_batch

from seq2seq_model import Seq2SeqModel

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 12, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('decode_batch_size', 80, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_decode_step', 500, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_string('model_path', None, 'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('decode_input', 'data/newstest2012.bpe.de', 'Decoding input path')
tf.app.flags.DEFINE_string('decode_output', 'data/newstest2012.bpe.de.trans', 'Decoding output path')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS

def load_config(FLAGS):
    
    config = util.unicode_to_utf8(
        json.load(open('%s.json' % FLAGS.model_path, 'rb')))
    for key, value in FLAGS.__flags.items():
        config[key] = value

    return config


def load_model(session, config):
    
    model = Seq2SeqModel(config, 'decode')
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print 'Reloading model parameters..'
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


def decode():
    # Load model config
    config = load_config(FLAGS)

    # Load source data to decode
    test_set = TextIterator(source=config['decode_input'],
                            batch_size=config['decode_batch_size'],
                            source_dict=config['source_vocabulary'],
                            maxlen=None,
                            n_words_source=config['num_encoder_symbols'])

    # Load inverse dictionary used in decoding
    target_inverse_dict = data_utils.load_inverse_dict(config['target_vocabulary'])
    
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Reload existing checkpoint
        model = load_model(sess, config)
        try:
            print 'Decoding {}..'.format(FLAGS.decode_input)
            if FLAGS.write_n_best:
                fout = [data_utils.fopen(("%s_%d" % (FLAGS.decode_output, k)), 'w') \
                        for k in range(FLAGS.beam_width)]
            else:
                fout = [data_utils.fopen(FLAGS.decode_output, 'w')]
            
            for idx, source_seq in enumerate(test_set):
                source, source_len = prepare_batch(source_seq)
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                predicted_ids = model.predict(sess, encoder_inputs=source, 
                                              encoder_inputs_length=source_len)
                   
                # Write decoding results
                for k, f in reversed(list(enumerate(fout))):
                    for seq in predicted_ids:
                        f.write(str(data_utils.seq2words(seq[:,k], target_inverse_dict)) + '\n')
                    if not FLAGS.write_n_best:
                        break
                print '  {}th line decoded'.format(idx * FLAGS.decode_batch_size)
                
            print 'Decoding terminated'
        except IOError:
            pass
        finally:
            [f.close() for f in fout]


def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run()

