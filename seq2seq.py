import numpy as np
import tensorflow as tf
import os
import sys
from tensorflow.contrib import learn
from collections import namedtuple


EncoderOutput = namedtuple("EncoderOutput", "output final_state sequence_len")
DecoderOutput = namedtuple("DecoderOutput", "logits cell_output")

tf.flags.DEFINE_string("source_file", "./data/france.txt", "source file")
tf.flags.DEFINE_string("target_file", "./data/english.txt", "target file")
tf.flags.DEFINE_integer("embedding_dim", 25, "demention of embedding")
tf.flags.DEFINE_integer("batch_size", 20, "batch size")
tf.flags.DEFINE_integer("num_epochs", 100, "number of epoch")
tf.flags.DEFINE_integer("checkpoint_every", 100, "save model every checkpoint step")

FLAGS = tf.flags.FLAGS

def load_source_and_target(source_file, target_file):
    source = []
    target = []
    with open(source_file, "r") as infile:
        source = infile.readlines()
    with open(target_file, "r") as infile:
        target = infile.readlines()
    return source, target

def preprocess():
    print("Loading data...")
    source, target = load_source_and_target(FLAGS.source_file, FLAGS.target_file)
    max_source_len = max([len(s.split()) for s in source])
    max_target_len = max([len(t.split()) for t in target])
    print max_source_len
    print max_target_len
    source_processor = learn.preprocessing.VocabularyProcessor(max_source_len)
    target_processor = learn.preprocessing.VocabularyProcessor(max_target_len)
    x = np.array(list(source_processor.fit_transform(source)))
    y = np.array(list(target_processor.fit_transform(target)))

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    print("Source Vocabulary Size: {:d}".format(len(source_processor.vocabulary_)))
    print("Target Vocabulary Size: {:d}".format(len(target_processor.vocabulary_)))

    return x_shuffled, y_shuffled, source_processor, target_processor, max_source_len, max_target_len

def gen_batch(x, y, batch_size, num_epochs, shuffle=True):
    x = np.array(x)
    batch_nums = x.shape[0] / batch_size
    data_size = x.shape[0]
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffle = x[shuffle_indices]
            y_shuffle = y[shuffle_indices]
        else:
            x_shuffle = x
            y_shuffle = y
        for i in range(batch_nums):
            start = i * batch_size
            end = (i+1) * batch_size
            batch_x = x_shuffle[start:end, :]
            batch_y = y_shuffle[start:end, :]
            sequence_len_x = np.sum(batch_x != 0, axis=1)
            sequence_len_y = np.sum(batch_y != 0, axis=1)
            yield batch_x, batch_y, sequence_len_x, sequence_len_y

#def encode(x, sequence_len):
#    cell_fw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_dim)
#    cell_bw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_dim)
#    output, state = tf.nn.bidirectional_dynamic_rnn(
#            cell_fw = cell_fw, 
#            cell_bw = cell_bw,
#            inputs=x,
#            sequence_length=sequence_len,
#            dtype=tf.float32)
#    output_cat = tf.concat(output, 2)
#    return EncoderOutput(output=output_cat,
#                         final_state=state,
#                         sequence_len=sequence_len)

def encode(x, sequence_len):
    with tf.variable_scope("encode"):
        cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_dim)
        output, state = tf.nn.dynamic_rnn(
                cell = cell, 
                inputs=x,
                sequence_length=sequence_len,
                dtype=tf.float32)
        return EncoderOutput(output=output,
                             final_state=state,
                             sequence_len=sequence_len)

def decode(input, sequence_len, initial_state, vocab_size):
    with tf.variable_scope("decode"):
        cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.embedding_dim)
        output, state = tf.nn.dynamic_rnn(cell=cell, inputs=input, 
                                          sequence_length=sequence_len,
                                          initial_state=initial_state)
        logits = tf.layers.dense(output, vocab_size)
        return DecoderOutput(logits=logits, cell_output=output)

def train(x_shuffled, y_shuffled, source_processor, target_processor, max_source_len, max_target_len):
    with tf.Session() as sess:
        input_x = tf.placeholder(name="x", shape=[None, max_source_len], dtype=tf.int32)
        input_y = tf.placeholder(name="y", shape=[None, max_target_len], dtype=tf.int32)
        x_len = tf.placeholder(name="x_len", dtype=tf.int32)
        y_len = tf.placeholder(name="y_len", dtype=tf.int32)
        source_emb = tf.get_variable(name="x_embedding", shape=[len(source_processor.vocabulary_), FLAGS.embedding_dim])
        x_emb = tf.nn.embedding_lookup(source_emb, input_x)
        target_emb = tf.get_variable(name="y_embedding", shape=[len(target_processor.vocabulary_), FLAGS.embedding_dim])
        y_emb = tf.nn.embedding_lookup(target_emb, input_y)

        encoder_output = encode(x_emb, x_len)
        decoder_output = decode(y_emb, y_len, encoder_output.final_state, len(target_processor.vocabulary_))
    
        #predict_ids = tf.expand_dims(input_x, 1)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_y, logits=decoder_output.logits)
        mask = tf.sequence_mask(y_len, max_target_len, dtype=tf.float32)
        loss = tf.reduce_mean(crossent * mask)
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        sess.run(tf.global_variables_initializer())

        batch_iter = gen_batch(x_shuffled, y_shuffled, FLAGS.batch_size, FLAGS.num_epochs)
        for batch in batch_iter:
            batch_x, batch_y, sequence_len_x, sequence_len_y = batch
            #print batch_x
            #print sequence_len_x
            #print batch_y
            #print sequence_len_y
            #res = sess.run(source_emb, feed_dict={input_x:batch_x, x_len:sequence_len_x})
            #en = sess.run(encoder_output, feed_dict={input_x:batch_x, x_len:sequence_len_x})
            #de = sess.run(decoder_output, feed_dict={input_x:batch_x, x_len:sequence_len_x, input_y:batch_y, y_len:sequence_len_y})
            #l = sess.run(loss, feed_dict={input_x:batch_x, x_len:sequence_len_x, input_y:batch_y, y_len:sequence_len_y})
            #print res.logits
            #print res.cell_output
            _, l = sess.run([train_op, loss], feed_dict={input_x:batch_x, x_len:sequence_len_x, input_y:batch_y, y_len:sequence_len_y})
            print l

def main(argv=None):
    x_shuffled, y_shuffled, source_processor, target_processor, max_source_len, max_target_len = preprocess()
    train(x_shuffled, y_shuffled, source_processor, target_processor, max_source_len, max_target_len)

if __name__ == "__main__":
    tf.app.run()

