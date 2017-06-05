# TF-seq2seq
## **Sequence to sequence (seq2seq) learning Using TensorFlow.**

The core building blocks are RNN Encoder-Decoder architectures and Attention mechanism.

The package was largely implemented by using the latest (1.2.rc1) tf.contrib.seq2seq modules
- AttentionWrapper
- Decoder
- BasicDecoder
- BeamSearchDecoder

**The package supports**
- Multi-layer
- Residual connection
- Dropout
- Attention and input_feeding
- Beamsearch decoding 
- Write n-best list
- **As of 1.2.rc1, the BeamSearchDecoder API works but does not produce correct outputs**

# Dependencies
- NumPy >= 1.11.1
- Tensorflow >= 1.2.rc1

# Usage Instructions
## **Data Preparation**

To preprocess raw parallel data of <code>sample_data.src</code> and <code>sample_data.trg</code>, simply run
```ruby
cd data/
./preprocess.sh src trg sample_data ${max_seq_len}
```

Running the above code performs widely used preprocessing steps for Machine Translation (MT).

- Normalizing punctuation
- Tokenizing
- Bytepair encoding (# merge = 30000) (Sennrich et al., 2016)
- Cleaning sequences of length over ${max_seq_len}
- Shuffling
- Building dictionaries

## **Training**
To train a seq2seq model,
```ruby
$ python seq2seq.py --cell_type='lstm' \ 
                    --attention_type='luong' \
                    --hidden_units=1024 \
                    --depth=2 \
                    --embedding_size=500 \
                    --num_encoder_symbols=30000 \
                    --num_decoder_symbols=30000 ...
```

## **Decoding**
To run the trained model for decoding,
```ruby
$ python seq2seq.py --decode \
                    --beam_width=5 \
                    --max_decode_step=300 \
                    --write_n_best
                    --decode_input=$PATH_TO_DECODE_INPUT
                    --decode_output=$PATH_TO_DECODE_OUTPUT
                    
```
If <code>--beam_width=1</code>, greedy decoding is performed at each time-step.

## **Arguments**

**Data params**
- <code>--source_vocabulary</code> : Path to source vocabulary
- <code>--target_vocabulary</code> : Path to target vocabulary
- <code>--source_train_data</code> : Path to source training data
- <code>--target_train_data</code> : Path to target training data
- <code>--source_valid_data</code> : Path to source validation data
- <code>--target_valid_data</code> : Path to target validation data

**Network params**
- <code>--cell_type</code> : RNN cell to use for encoder and decoder (default: lstm)
- <code>--attention_type</code> : Attention mechanism (bahdanau, luong), (default: bahdanau)
- <code>--depth</code> : Number of hidden units for each layer in the model (default: 2)
- <code>--embedding_size</code> : Embedding dimensions of encoder and decoder inputs (default: 500)
- <code>--num_encoder_symbols</code> : Source vocabulary size to use (default: 30000)
- <code>--num_decoder_symbols</code> : Target vocabulary size to use (default: 30000)
- <code>--use_residual</code> : Number of hidden units for each layer in the model (default: True)
- <code>--attn_input_feeding</code> : Use input feeding method in attentional decoder (Luong et al., 2015) (default: True)
- <code>--use_dropout</code> : Use dropout in each rnn cell (default: True)
- <code>--dropout_rate</code> : Dropout probability for cell outputs (0.0: no dropout) (default: 0.3)

**Training params**
- <code>--learning_rate</code> : Number of hidden units for each layer in the model (default: 0.0002)
- <code>--max_gradient_norm</code> : Clip gradients to this norm (default 1.0)
- <code>--batch_size</code> : Batch size
- <code>--max_epochs</code> : Maximum training epochs
- <code>--max_load_batches</code> : Maximum number of batches to load at one time
- <code>--max_seq_length</code> : Maximum sequence length
- <code>--display_freq</code> : Display training status every this iteration
- <code>--save_freq</code> : Save model checkpoint every this iteration
- <code>--valid_freq</code> : Evaluate the model every this iteration: valid_data needed
- <code>--optimizer</code> : Optimizer for training: (adadelta, adam, rmsprop) (default: adam)
- <code>--model_dir</code> : Path to save model checkpoints
- <code>--model_name</code> : File name used for model checkpoints
- <code>--shuffle_each_epoch</code> : Shuffle training dataset for each epoch (default: True)
- <code>--sort_by_length</code> : Sort pre-fetched minibatches by their target sequence lengths (default: True)

**Decoding params**
- <code>--decode</code> : Use decode mode (default: False)
- <code>--beam_width</code> : Beam width used in beamsearch (default: 1)
- <code>--max_decode_step</code> : Maximum time step limit in decoding (default: 500)
- <code>--write_n_best</code> : Write beamsearch n-best list (n=beam_width) (default: False)
- <code>--decode_input</code> : Input file path to decode
- <code>--decode_output</code> : Output file path of decoding output

**Runtime params**
- <code>--allow_soft_placement</code> : Allow device soft placement
- <code>--log_device_placement</code> : Log placement of ops on devices


## History
- June 5, 2017: Major update. Resolved incompatibility issue between BeamSearchDecoder and AttentionWrapper with no predefined input time steps.


## Acknowledgements

The implementation is based on following projects:
- [nematus](https://github.com/rsennrich/nematus/): Theano implementation of Neural Machine Translation
- [tf.seq2seq_legacy](https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate) Legacy Tensorflow seq2seq tutorial
- [tf_tutorial_plus](https://github.com/j-min/tf_tutorial_plus): Nice tutorials for tf.contrib.seq2seq API

For any comments and feedbacks, please email me at pjh0308@gmail.com or open an issue here.
