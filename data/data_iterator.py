
import numpy as np
import shuffle
from util import load_dict

import data_utils

'''
Much of this code is based on the data_iterator.py of
nematus project (https://github.com/rsennrich/nematus)
'''

class TextIterator:
    """Simple Text iterator."""
    def __init__(self, source, source_dict,
                 batch_size=128, maxlen=None,
                 n_words_source=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 maxibatch_size=20,
                 ):

        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main([self.source_orig], temporary=True)
        else:
            self.source = data_utils.fopen(source, 'r')

        self.source_dict = load_dict(source_dict)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * maxibatch_size
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source = shuffle.main([self.source_orig], temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []

        # fill buffer, if it's empty
        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip().split())
    
            # sort by buffer
            if self.sort_by_length:
                slen = np.array([len(s) for s in self.source_buffer])
                sidx = slen.argsort()
    
                _sbuf = [self.source_buffer[i] for i in sidx]
    
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()
    
        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
    
        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict
                      else data_utils.unk_token for w in ss]
    
                if self.maxlen and len(ss) > self.maxlen:
                    continue
                if self.skip_empty and (not ss):
                    continue
                source.append(ss)
    
                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
    
        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0:
            source = self.next()
    
        return source

class BiTextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = data_utils.fopen(source, 'r')
            self.target = data_utils.fopen(target, 'r')

        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]

        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict
                      else data_utils.unk_token for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict 
                      else data_utils.unk_token for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target 
                          else data_utils.unk_token for w in tt]

                if self.maxlen:
                    if len(ss) > self.maxlen and len(tt) > self.maxlen:
                        continue
                if self.skip_empty and (not ss or not tt):
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target
