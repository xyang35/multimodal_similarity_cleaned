import tensorflow as tf
import numpy as np
import utils
import functools
from tensorflow.python.ops.rnn import _transpose_batch_time


class Seq2seqTSN(object):
    """
    Sequence to sequence model for initializeing sequence representation
    """

    def name(self):
        return "Seq2seqTSN"

    def __init__(self, n_seg, n_input=8, emb_dim=128, reverse=False):
        self.n_seg = n_seg
        self.n_input = n_input
        self.emb_dim = emb_dim
        self.reverse = reverse

        self.prepare_input = functools.partial(utils.tsn_prepare_input, self.n_seg)
        self.prepare_input_test = functools.partial(utils.tsn_prepare_input_test, self.n_seg)
        self.prepare_input_tf = functools.partial(utils.tsn_prepare_input_tf, self.n_seg)

        with tf.variable_scope("Seq2seqTSN"):
            self.W_encode = tf.get_variable(name="W_encode", shape=[self.n_input, self.emb_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_encode = tf.get_variable(name="b_encode", shape=[self.emb_dim],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            # for decoding
            self.W_decode1 = tf.get_variable(name="W_decode1", shape=[self.emb_dim, self.emb_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_decode1 = tf.get_variable(name="b_decode1", shape=[self.emb_dim],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            self.b_decode2 = tf.get_variable(name="b_decode2", shape=[self.n_input],
                                initializer=tf.zeros_initializer(),
                                trainable=True)

            with tf.variable_scope("encoder"):
                self.encoder_cell = tf.contrib.rnn.LSTMCell(self.emb_dim, forget_bias=1.0)
            with tf.variable_scope("decoder"):
                self.decoder_cell = tf.contrib.rnn.LSTMCell(self.emb_dim, forget_bias=1.0)

    def forward(self, x, keep_prob):
        """
        x -- input features, [batch_size, n_seg, n_input]
        """
        
        if self.reverse:
            # reverse the sequence if needed, claimed to be useful for NMT
            x = x[:, ::-1, :]

        batch_size = tf.shape(x)[0]
        seq_len = tf.ones((batch_size,), dtype='int32') * self.n_seg

        ###################### Encoder ###################

        def RNN(x):
            dropout_cell = tf.contrib.rnn.DropoutWrapper(self.encoder_cell, input_keep_prob=keep_prob)    # onlyt input dropout is used
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    dropout_cell, x, seq_len, dtype=tf.float32, scope="Seq2seqTSN/encoder")
            return encoder_outputs[:, -1], encoder_final_state

        # encode
        x_flat = tf.reshape(x, [-1, self.n_input])
        h_encode = tf.nn.relu(tf.nn.xw_plus_b(x_flat, self.W_encode, self.b_encode))
        h_encode = tf.reshape(h_encode, [-1, self.n_seg, self.emb_dim])

        self.hidden, encoder_final_state = RNN(h_encode)

        ###################### Decoder ###################

        def loop_fn(time, cell_output, cell_state, loop_state):
            def get_next_input():
                if cell_state is None:
                    next_input = tf.zeros([batch_size, self.n_input], dtype=tf.float32)
                else:
                    #next_input = tf.nn.xw_plus_b(cell_output, self.W_ho, self.b_o)   # conditioned
                    next_input = tf.zeros([batch_size, self.n_input], dtype=tf.float32)    # un-conditioned
                return next_input
                
            emit_output = cell_output

            if cell_state is None:
                next_cell_state = encoder_final_state
            else:
                next_cell_state = cell_state

            elements_finished = (time >= seq_len)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                        finished,
                        lambda: tf.zeros([batch_size, self.n_input], dtype=tf.float32),
                        get_next_input)
            next_loop_state = None

            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        # decode
        outputs_ta, final_state, _ = tf.nn.raw_rnn(self.decoder_cell, loop_fn, scope="Seq2seqTSN/decoder")
        outputs = _transpose_batch_time(outputs_ta.stack())    # outputs and shape [batch_size, time ,output_dim]

        outputs = tf.reshape(outputs, [-1, self.emb_dim])
        h_decode = tf.nn.relu(tf.nn.xw_plus_b(outputs, self.W_decode1, self.b_decode1))

        x_recon = tf.nn.xw_plus_b(h_decode, tf.transpose(self.W_encode), self.b_decode2)
        self.x_recon = tf.reshape(x_recon, [-1, self.n_seg, self.n_input])



class SAE(object):
    """
    Stack autoencder for initializing data representation

    Use tied weights
    Denoising is not used because we don't have layer-wise pretraining
    """

    def name(self):
        return "SAE"

    def __init__(self, n_input=8, emb_dim=128):
        self.n_input = n_input
        self.emb_dim = emb_dim

        self.W_1 = tf.get_variable(name="W_1", shape=[self.n_input, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_1 = tf.get_variable(name="b_1", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_2 = tf.get_variable(name="W_2", shape=[self.emb_dim, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_2 = tf.get_variable(name="b_2", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        # for decoding
        self.b_3 = tf.get_variable(name="b_3", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.b_4 = tf.get_variable(name="b_4", shape=[self.n_input],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def forward(self, x):

        # encode
        h = tf.nn.relu(tf.nn.xw_plus_b(x, self.W_1, self.b_1))
        self.hidden = tf.nn.xw_plus_b(h, self.W_2, self.b_2)

        # decode
        h_recon = tf.nn.relu(tf.nn.xw_plus_b(self.hidden, tf.transpose(self.W_2), self.b_3))
        self.x_recon = tf.nn.xw_plus_b(h_recon, tf.transpose(self.W_1), self.b_4)

class PairSim2(object):
    """ PairSim layer
    Input a pair of sample A, B
    Ouput a binary classification: whether A, B is similar with each other

    One choice: first calculate distance (A-B)^2 then do prediction (reference: A Discriminatively Learned CNN Embedding for Person Re-identification)
    Or: concatenate two features then do prediction (reference: A Multi-Task Deep Network for Person Re-Identification)

    We choose the first one here.
    """

    def name(self):
        return "PairSim"

    def __init__(self, n_input=128):
        self.n_input = n_input

        with tf.variable_scope("PairSim"):
            self.W_pairwise = tf.get_variable(name="W_pairwise", shape=[self.n_input, self.n_input],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_pairwise = tf.get_variable(name="b_pairwise", shape=[self.n_input],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            self.W_o = tf.get_variable(name="W_o", shape=[self.n_input, 2],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_o = tf.get_variable(name="b_o", shape=[2],
                                initializer=tf.zeros_initializer(),
                                trainable=True)

    def forward(self, x, keep_prob):
        """
        x -- feature pair, [batch_size, 2, n_input]
        """

        x_A, x_B = tf.unstack(x, 2, 1)
        x_diff = tf.square(x_A - x_B)

        h = tf.nn.relu(tf.nn.xw_plus_b(x_diff, self.W_pairwise, self.b_pairwise))
        h_drop = tf.nn.dropout(h, keep_prob)

        self.logits = tf.nn.xw_plus_b(h_drop, self.W_o, self.b_o)    # for computing loss
        self.prob = tf.nn.softmax(self.logits)

class PairSim(object):
    """ PairSim layer
    Input a pair of sample A, B
    Ouput a binary classification: whether A, B is similar with each other

    One choice: first calculate distance (A-B)^2 then do prediction (reference: A Discriminatively Learned CNN Embedding for Person Re-identification)
    Or: concatenate two features then do prediction (reference: A Multi-Task Deep Network for Person Re-Identification)

    We choose the later one here.
    """

    def name(self):
        return "PairSim"

    def __init__(self, n_input=128):
        self.n_input = n_input

        with tf.variable_scope("PairSim"):
            self.W_pairwise = tf.get_variable(name="W_pairwise", shape=[self.n_input*2, self.n_input],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_pairwise = tf.get_variable(name="b_pairwise", shape=[self.n_input],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            self.W_o = tf.get_variable(name="W_o", shape=[self.n_input, 2],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_o = tf.get_variable(name="b_o", shape=[2],
                                initializer=tf.zeros_initializer(),
                                trainable=True)

    def forward(self, x, keep_prob):
        """
        x -- feature pair, [batch_size, 2, n_input]
        """

        x_concat = tf.reshape(x, [-1, 2*self.n_input])
        x_drop = tf.nn.dropout(x_concat, keep_prob)

        h = tf.nn.relu(tf.nn.xw_plus_b(x_drop, self.W_pairwise, self.b_pairwise))
        h_drop = tf.nn.dropout(h, keep_prob)

        self.logits = tf.nn.xw_plus_b(h_drop, self.W_o, self.b_o)    # for computing loss
        self.prob = tf.nn.softmax(self.logits)

class PDDM(object):
    """ PDDM layer
    Input a pair of sample A, B
    Ouput a score with 0 to 1: whether A, B is similar with each other

    reference: Local Similarity-Aware Deep Feature Embedding
    """

    def name(self):
        return "PDDM"

    def __init__(self, n_input=128):
        self.n_input = n_input

        with tf.variable_scope("PDDM"):
            self.W_u = tf.get_variable(name="W_u", shape=[self.n_input, self.n_input],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_u = tf.get_variable(name="b_u", shape=[self.n_input],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            self.W_v = tf.get_variable(name="W_v", shape=[self.n_input, self.n_input],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_v = tf.get_variable(name="b_v", shape=[self.n_input],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            self.W_c = tf.get_variable(name="W_c", shape=[2*self.n_input, self.n_input],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_c = tf.get_variable(name="b_c", shape=[self.n_input],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            self.W_s = tf.get_variable(name="W_s", shape=[self.n_input, 2],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_s = tf.get_variable(name="b_s", shape=[2],
                                initializer=tf.zeros_initializer(),
                                trainable=True)

    def forward(self, x):
        """
        x -- feature pair, [batch_size, 2, n_input]
        """

        x_i, x_j = tf.unstack(x, 2, 1)

        u = tf.abs(x_i - x_j)
        v = 0.5 * (x_i + x_j)

        uu = tf.nn.l2_normalize(tf.nn.relu(tf.nn.xw_plus_b(u, self.W_u, self.b_u)), axis=-1,epsilon=1e-10)
        vv = tf.nn.l2_normalize(tf.nn.relu(tf.nn.xw_plus_b(v, self.W_v, self.b_v)), axis=-1,epsilon=1e-10)

        c = tf.nn.relu(tf.nn.xw_plus_b(tf.concat((uu,vv), axis=1), self.W_c, self.b_c))
        self.logits = tf.nn.xw_plus_b(c, self.W_s, self.b_s)
        self.prob = tf.nn.softmax(self.logits)


class OutputLayer(object):
    def name(self):
        return "OutputLayer"

    def __init__(self, n_input, n_output):

        self.n_input = n_input
        self.n_output = n_output

        with tf.variable_scope("OutputLayer"):
            self.W = tf.get_variable(name="W", shape=[self.n_input, self.n_output],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b = tf.get_variable(name="b", shape=[self.n_output],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            self.W_o = tf.get_variable(name="W_o", shape=[self.n_output, self.n_output],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b_o = tf.get_variable(name="b_o", shape=[self.n_output],
                                initializer=tf.zeros_initializer(),
                                trainable=True)

    def forward(self, x, keep_prob):
        """
        x -- feature batch, [batch_size, n_input]
        """

        hidden = tf.nn.xw_plus_b(x, self.W, self.b)
        hidden_drop = tf.nn.dropout(tf.nn.relu(hidden), keep_prob)
        self.logits = tf.nn.xw_plus_b(hidden_drop, self.W_o, self.b_o)

class CUBLayer(object):
    def name(self):
        return "CUBLayer"

    def __init__(self, n_input, n_output):

        self.n_input = n_input
        self.n_output = n_output

        with tf.variable_scope("CUBLayer"):
            self.W = tf.get_variable(name="W", shape=[self.n_input, self.n_output],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1.),
                                trainable=True)
            self.b = tf.get_variable(name="b", shape=[self.n_output],
                                initializer=tf.zeros_initializer(),
                                trainable=True)

    def forward(self, x, keep_prob):
        """
        x -- feature batch, [batch_size, n_input]
        """

        x_drop = tf.nn.dropout(x, keep_prob)
        self.logits = tf.nn.xw_plus_b(x_drop, self.W, self.b)

# Recurrent TSN
class RTSN(object):
    def name(self):
        return "RTSN"

    def __init__(self, n_seg=3, emb_dim=128, n_input=8):
        
        self.n_seg = n_seg
        self.n_input = n_input
        self.emb_dim = emb_dim

        self.prepare_input = functools.partial(utils.tsn_prepare_input, self.n_seg)
        self.prepare_input_test = functools.partial(utils.tsn_prepare_input_test, self.n_seg)


        with tf.variable_scope("RTSN"):
            self.W_1 = tf.get_variable(name="W_1", shape=[self.n_input, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
            self.b_1 = tf.get_variable(name="b_1", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
            self.encoder_cell = tf.contrib.rnn.LSTMCell(self.emb_dim, forget_bias=1.0)

    def forward(self, x, keep_prob):
        """
        x -- input features, [batch_size, n_seg, n_input]
        """

        def RNN(x):
            dropout_cell = tf.contrib.rnn.DropoutWrapper(self.encoder_cell, input_keep_prob=keep_prob)    # onlyt input dropout is used
            seq_len = tf.ones((tf.shape(x)[0],), dtype='int32') * self.n_seg
            encoder_outputs, _ = tf.nn.dynamic_rnn(dropout_cell, x, seq_len, dtype=tf.float32, scope="RTSN")
            return encoder_outputs[:, -1]

        x_flat = tf.reshape(x, [-1, self.n_input])
        h1 = tf.nn.relu(tf.nn.xw_plus_b(x_flat, self.W_1, self.b_1))

        h1 = tf.reshape(h1, [-1, self.n_seg, self.emb_dim])
        self.hidden = RNN(h1)

# TSN for temporal aggregation
class TSN(object):
    def name(self):
        return "TSN"

    def __init__(self, n_seg=3, emb_dim=128, n_input=8):
        
        self.n_seg = n_seg
        self.n_input = n_input
        self.emb_dim = emb_dim

        self.prepare_input = functools.partial(utils.tsn_prepare_input, self.n_seg)
        self.prepare_input_test = functools.partial(utils.tsn_prepare_input_test, self.n_seg)

        self.W_1 = tf.get_variable(name="W_1", shape=[self.n_input, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_1 = tf.get_variable(name="b_1", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_2 = tf.get_variable(name="W_2", shape=[self.emb_dim, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_2 = tf.get_variable(name="b_2", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def forward(self, x, keep_prob):
        """
        x -- input features, [batch_size, n_seg, n_input]
        """

        x_flat = tf.reshape(x, [-1, self.n_input])
        h1 = tf.nn.relu(tf.nn.xw_plus_b(x_flat, self.W_1, self.b_1))
        h1_drop = tf.nn.dropout(h1, keep_prob)

        h2 = tf.nn.xw_plus_b(h1_drop, self.W_2, self.b_2)
        h2_reshape = tf.reshape(h2, [-1, self.n_seg, self.emb_dim])

        self.hidden = tf.reduce_mean(h2_reshape, axis=1)

# Bidirectional Recurrent convolutional TSN
class ConvBiRTSN(object):
    def name(self):
        return "ConvBiRTSN"

    def __init__(self, n_seg=3, n_input=1536, n_h=8, n_w=8, n_C=20, emb_dim=128):

        self.n_seg = n_seg
        self.n_C = n_C
        self.n_h = n_h
        self.n_w = n_w
        self.n_input = n_input
        self.emb_dim = emb_dim

        self.prepare_input = functools.partial(utils.tsn_prepare_input, self.n_seg)
        self.prepare_input_test = functools.partial(utils.tsn_prepare_input_test, self.n_seg)
        self.prepare_input_tf = functools.partial(utils.tsn_prepare_input_tf, self.n_seg)

        with tf.variable_scope("ConvBiRTSN"):
            self.W_emb = tf.get_variable(name="W_emb", shape=[1,1,n_input,self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
            # forward pass
            self.fw_cell = tf.contrib.rnn.LSTMCell(self.emb_dim//2, forget_bias=1.0)
            self.bw_cell = tf.contrib.rnn.LSTMCell(self.emb_dim//2, forget_bias=1.0)

    def forward(self, x, keep_prob):
        """
        x -- input features, [batch_size, n_seg, n_h, n_w, n_input]
        """

        def RNN(x):
            fw_dropout_cell = tf.contrib.rnn.DropoutWrapper(self.fw_cell, input_keep_prob=keep_prob)    # onlyt input dropout is used
            bw_dropout_cell = tf.contrib.rnn.DropoutWrapper(self.bw_cell, input_keep_prob=keep_prob)    # onlyt input dropout is used
            seq_len = tf.ones((tf.shape(x)[0],), dtype='int32') * self.n_seg

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_dropout_cell, bw_dropout_cell, x, seq_len, dtype=tf.float32, scope="ConvBiRTSN")

            # concatenate outputs
            encoder_outputs = tf.concat(outputs, 2)
            return encoder_outputs[:, -1]

        x_flat = tf.reshape(x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1, 1, 1, 1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [-1, self.n_seg, self.n_h*self.n_w*self.n_C])
        self.hidden = RNN(x_emb)


# Recurrent convolutional TSN
class ConvRTSN(object):
    def name(self):
        return "ConvRTSN"

    def __init__(self, n_seg=3, n_input=1536, n_h=8, n_w=8, n_C=20, emb_dim=128):

        self.n_seg = n_seg
        self.n_C = n_C
        self.n_h = n_h
        self.n_w = n_w
        self.n_input = n_input
        self.emb_dim = emb_dim

        self.prepare_input = functools.partial(utils.tsn_prepare_input, self.n_seg)
        self.prepare_input_test = functools.partial(utils.tsn_prepare_input_test, self.n_seg)
        self.prepare_input_tf = functools.partial(utils.tsn_prepare_input_tf, self.n_seg)

        with tf.variable_scope("ConvRTSN"):
            self.W_emb = tf.get_variable(name="W_emb", shape=[1,1,n_input,self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
            self.encoder_cell = tf.contrib.rnn.LSTMCell(self.emb_dim, forget_bias=1.0)

    def forward(self, x, keep_prob):
        """
        x -- input features, [batch_size, n_seg, n_h, n_w, n_input]
        """

        def RNN(x):
            dropout_cell = tf.contrib.rnn.DropoutWrapper(self.encoder_cell, input_keep_prob=keep_prob)    # onlyt input dropout is used
            seq_len = tf.ones((tf.shape(x)[0],), dtype='int32') * self.n_seg
            encoder_outputs, _ = tf.nn.dynamic_rnn(dropout_cell, x, seq_len, dtype=tf.float32, scope="ConvRTSN")
            return encoder_outputs[:, -1]

        x_flat = tf.reshape(x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1, 1, 1, 1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [-1, self.n_seg, self.n_h*self.n_w*self.n_C])
        self.hidden = RNN(x_emb)


# Convolutional embedding + LSTM for sequence encoding
class ConvLSTM(object):
    def name(self):
        return "ConvLSTM"

    def __init__(self, max_time, n_input=1536, n_h=8, n_w=8, n_C=20, emb_dim=128):

        self.max_time = max_time
        self.n_C = n_C
        self.n_h = n_h
        self.n_w = n_w
        self.n_input = n_input
        self.emb_dim = emb_dim

        self.prepare_input = functools.partial(utils.rnn_prepare_input, self.max_time)

        with tf.variable_scope("ConvLSTM"):
            self.W_emb = tf.get_variable(name="W_emb", shape=[1,1,n_input,self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
            self.encoder_cell = tf.contrib.rnn.LSTMCell(self.emb_dim, forget_bias=1.0)

    def forward(self, x, seq_len):
        """
        Argument:
            x -- input features, [batch_size, max_time, n_h, n_w, n_input]
            seq_len -- length indicator, [batch_size, ]
        """

        batch_size = tf.shape(x)[0]

        def RNN(x, seq_len):
            encoder_outputs, _ = tf.nn.dynamic_rnn(self.encoder_cell, x, seq_len, dtype=tf.float32, scope="ConvLSTM")

            # slice the valid output
            indices = tf.stack([tf.range(batch_size), seq_len-1], axis=1)
            return tf.gather_nd(encoder_outputs, indices)

        x_flat = tf.reshape(x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1, 1, 1, 1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [-1, self.max_time, self.n_h*self.n_w*self.n_C])
        self.hidden = RNN(x_emb, seq_len)



# Convolutional embedding + TSN for temporal aggregation
class ConvTSN(object):
    def name(self):
        return "ConvTSN"

    def __init__(self, n_seg=3, n_C=20, emb_dim=256, n_input=1536, n_h=8, n_w=8, n_output=11):
        
        self.n_seg = n_seg
        self.n_C = n_C
        self.n_h = n_h
        self.n_w = n_w
        self.n_input = n_input
        self.n_output = n_output
        self.emb_dim = emb_dim

        self.prepare_input = functools.partial(utils.tsn_prepare_input, self.n_seg)
        self.prepare_input_test = functools.partial(utils.tsn_prepare_input_test, self.n_seg)

        self.W_emb = tf.get_variable(name="W_emb", shape=[1,1,n_input,self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.W = tf.get_variable(name="W", shape=[self.n_C*n_h*n_w, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b = tf.get_variable(name="b", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def forward(self, x, keep_prob):
        """
        x -- input features, [batch_size, n_seg, n_h, n_w, n_input]
        """

        x_flat = tf.reshape(x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1,1,1,1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [-1, self.n_h*self.n_w*self.n_C])
        h = tf.nn.xw_plus_b(x_emb, self.W, self.b)

        h_reshape = tf.reshape(h, [-1, self.n_seg, self.emb_dim])

        self.hidden = tf.reduce_mean(h_reshape, axis=1)



# Convolutional TSN for classification
class ConvTSNClassifier(object):
    def name(self):
        return "ConvTSN"

    def __init__(self, n_seg=3, n_C=20, emb_dim=256, input_keep_prob=1.0, output_keep_prob=1.0, n_input=1536, n_h=8, n_w=8, n_output=11):
        
        self.n_seg = n_seg
        self.n_C = n_C
        self.n_h = n_h
        self.n_w = n_w
        self.n_input = n_input
        self.n_output = n_output
        self.emb_dim = emb_dim
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob

        self.prepare_input = functools.partial(utils.tsn_prepare_input, self.n_seg)
        self.prepare_input_test = functools.partial(utils.tsn_prepare_input_test, self.n_seg)

        self.W_emb = tf.get_variable(name="W_emb", shape=[1,1,n_input,self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.W = tf.get_variable(name="W", shape=[self.n_C*n_h*n_w, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b = tf.get_variable(name="b", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_o = tf.get_variable(name="W_o", shape=[self.emb_dim, self.n_output],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_o = tf.get_variable(name="b_o", shape=[self.n_output],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def forward(self, x):
        """
        x -- input features, [batch_size, n_seg, n_h, n_w, n_input]
        """

        x_flat = tf.reshape(x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1,1,1,1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [-1, self.n_h*self.n_w*self.n_C])
        h = tf.nn.xw_plus_b(x_emb, self.W, self.b)

        h_reshape = tf.reshape(h, [-1, self.n_seg, self.emb_dim])
        self.feat = tf.reduce_mean(h_reshape, axis=1)

        h_drop = tf.nn.dropout(tf.nn.relu(h), self.output_keep_prob)
        output = tf.nn.xw_plus_b(h_drop, self.W_o, self.b_o)
        output_reshape = tf.reshape(output, [-1, self.n_seg, self.n_output])

        self.logits = tf.reduce_mean(output_reshape, axis=1)



# triplet loss
def triplet_loss(anchor, positive, negative, alpha=0.2):

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # FIXME: use softplus?
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)


# weighted triplet loss
def weighted_triplet_loss(anchor, positive, negative, prob_pos, prob_neg, alpha=0.2):
    """
    prob_pos -- tf.float32, [N,] similarity confidence of anchor-positive pairs
    prob_neg -- tf.float32, [N,] similarity confidence of anchor-negative pairs

    4-class problem:
        § p1(1-p2)L(A,B,C)
        § (1-p1)p2L(A,C,B)
        § p1p2[L(A,B,A)+L(A,C,A)]/2
        § (1-p1)(1-p2)[L(A,A,B)+L(A,A,C)]/2
    """

    def _triplet_loss(anc, pos, neg, alpha):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anc, pos)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anc, neg)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # FIXME: use softplus?
        return tf.maximum(basic_loss, 0.0)

    w1 = tf.multiply(prob_pos, (1-prob_neg))
    w2 = tf.multiply((1-prob_pos), prob_neg)
    w3 = tf.multiply(prob_pos, prob_neg)
    w4 = tf.multiply((1-prob_pos), (1-prob_neg))

    weighted_loss = tf.multiply(w1, _triplet_loss(anchor, positive, negative, alpha)) + \
                    tf.multiply(w2, _triplet_loss(anchor, negative, positive, alpha)) + \
                    tf.multiply(w3, 0.5*(_triplet_loss(anchor, positive, anchor, -alpha*2)+_triplet_loss(anchor, negative, anchor, -alpha*2))) + \
                    tf.multiply(w4, 0.5*(_triplet_loss(anchor, anchor, positive, alpha*2)+_triplet_loss(anchor, anchor, negative, alpha*2)))

    return tf.reduce_mean(weighted_loss, 0), tf.stack([w1,w2,w3,w4], axis=1)

def weighted_triplet_loss(anchor, positive, negative, prob_pos, prob_neg, alpha=0.2):
    """
    prob_pos -- tf.float32, [N,] similarity confidence of anchor-positive pairs
    prob_neg -- tf.float32, [N,] similarity confidence of anchor-negative pairs

    4-class problem:
        § p1(1-p2)L(A,B,C)
        § (1-p1)p2L(A,C,B)
        § p1p2[L(A,B,A)+L(A,C,A)]/2
        § (1-p1)(1-p2)[L(A,A,B)+L(A,A,C)]/2
    """

    def _triplet_loss(anc, pos, neg, alpha):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anc, pos)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anc, neg)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # FIXME: use softplus?
        return tf.maximum(basic_loss, 0.0)

    w1 = tf.multiply(prob_pos, (1-prob_neg))
    w2 = tf.multiply((1-prob_pos), prob_neg)
    w3 = tf.multiply(prob_pos, prob_neg)
    w4 = tf.multiply((1-prob_pos), (1-prob_neg))

    weighted_loss = tf.multiply(w1, _triplet_loss(anchor, positive, negative, alpha)) + \
                    tf.multiply(w2, _triplet_loss(anchor, negative, positive, alpha)) + \
                    tf.multiply(w3, 0.5*(_triplet_loss(anchor, positive, anchor, -alpha*2)+_triplet_loss(anchor, negative, anchor, -alpha*2))) + \
                    tf.multiply(w4, 0.5*(_triplet_loss(anchor, anchor, positive, alpha*2)+_triplet_loss(anchor, anchor, negative, alpha*2)))

    return tf.reduce_mean(weighted_loss, 0), tf.stack([w1,w2,w3,w4], axis=1)

# Weighted Batch-hard loss
# reference: In Defense of the Triplet Loss for Person Re-Identification
# (https://github.com/VisualComputingInstitute/triplet-reid/blob/master/loss.py)
def batch_hard(dists, pids, margin, weighted=True):

    with tf.name_scope("batch_hard"):
        batch_size = tf.cast(tf.shape(dists)[0], tf.float32)

        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists*tf.cast(positive_mask, tf.float32), axis=1)
        closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
                                    (dists, negative_mask), tf.float32)

        diff = furthest_positive - closest_negative
        if margin == "soft":
            diff = tf.nn.softplus(diff)
        else:
            diff = tf.maximum(diff + margin, 0.0)

        if weighted:
            # reweight the losses, inversely proportional to class frequencies
            # also mask out background class as anchor
            foreground_mask = tf.not_equal(pids, 0.0)
            foreground_num = tf.reduce_sum(tf.cast(foreground_mask, tf.float32))

            weights = tf.reduce_sum(tf.cast(negative_mask, tf.float32), axis=1)
            weights = tf.multiply(weights, tf.cast(foreground_mask, tf.float32))    # only count foreground
            weights = tf.divide(weights, tf.reduce_sum(weights))
        else:
            weights = tf.divide(1.0, batch_size)

        loss = tf.reduce_sum(tf.multiply(diff, weights))   # weighted loss
        num_active = tf.reduce_sum(tf.cast(tf.greater(diff*tf.cast(foreground_mask,tf.float32), 1e-5), tf.float32)) / foreground_num

    return loss, num_active, diff, weights, furthest_positive, closest_negative

def lifted_loss(dists, pids, margin, weighted=True):

    with tf.name_scope("lifted_loss"):
        batch_size = tf.cast(tf.shape(dists)[0], tf.float32)

        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_logsumexp(dists*tf.cast(positive_mask, tf.float32), axis=1)
        closest_negative = tf.map_fn(lambda x: tf.reduce_logsumexp(tf.boolean_mask(x[0], x[1])),
                                    (margin-dists, negative_mask), tf.float32)

        diff = furthest_positive + closest_negative
#        diff = tf.nn.softplus(diff)
        diff = tf.maximum(diff, 0.0)

        if weighted:
            # reweight the losses, inversely proportional to class frequencies
            # also mask out background class as anchor
            foreground_mask = tf.not_equal(pids, 0.0)
            foreground_num = tf.reduce_sum(tf.cast(foreground_mask, tf.float32))

            weights = tf.reduce_sum(tf.cast(negative_mask, tf.float32), axis=1)
            weights = tf.multiply(weights, tf.cast(foreground_mask, tf.float32))    # only count foreground
            weights = tf.divide(weights, tf.reduce_sum(weights))
        else:
            weights = tf.divide(1.0, batch_size)

        loss = tf.reduce_sum(tf.multiply(diff, weights))   # weighted loss
#        num_active = tf.reduce_sum(tf.cast(tf.greater(diff*tf.cast(foreground_mask,tf.float32), 1e-5), tf.float32)) / foreground_num
        num_active = 1.0

    return loss, num_active, diff, weights, furthest_positive, closest_negative


# Deep CCA loss, reference: On Deep Multi-view Representation Learning
def dcca_loss(X1, X2, K=0, rcov1=1e-4, rcov2=1e-4):
    """
    X1 / X2: network output for view 1 / 2, shape: [N, dim1 or dim2]
    K:  dimensionality of CCA projection
    rcov1 / rcov2: optional regularization parameter for view 1/2

    Tested in ../preprocess/script.py, works similarly as cca in sklearn
    """

    N = tf.cast(tf.shape(X1)[0], dtype=tf.float32)
    d1 = X1.get_shape().as_list()[1]
    d2 = X2.get_shape().as_list()[1]
    if K == 0:
        K = min(d1, d2)

    # remove mean
    X1 -= tf.reduce_mean(X1, axis=0, keepdims=True) 
    X2 -= tf.reduce_mean(X2, axis=0, keepdims=True) 

    S11 = tf.matmul(tf.transpose(X1), X1) / (N-1) + rcov1 * tf.eye(d1, dtype=X1.dtype)
    S22 = tf.matmul(tf.transpose(X2), X2) / (N-1) + rcov2 * tf.eye(d2, dtype=X2.dtype)
    S12 = tf.matmul(tf.transpose(X1), X2) / (N-1)
    D1, V1 = tf.self_adjoint_eig(S11)
    D2, V2 = tf.self_adjoint_eig(S22)
    # for numerical stability
    # The coordinates are returned in a 2-D tensor where the first dimension (rows) represents the number of true elements, and the second dimension (columns) represents the coordinates of the true elements.
    idx1 = tf.squeeze(tf.where(tf.greater(D1, 1e-12)))
    D1 = tf.gather(D1, idx1)
    V1 = tf.gather(V1, idx1, axis=1)
    idx2 = tf.squeeze(tf.where(tf.greater(D2, 1e-12)))
    D2 = tf.gather(D2, idx2)
    V2 = tf.gather(V2, idx2, axis=1)

    K11 = tf.matmul(tf.matmul(V1, tf.diag(tf.pow(D1, -0.5))), tf.transpose(V1))
    K22 = tf.matmul(tf.matmul(V2, tf.diag(tf.pow(D2, -0.5))), tf.transpose(V2))
    T = tf.reshape(tf.matmul(tf.matmul(K11, S12), K22), [d1,d2])    # use reshape to prevent error: SVD gradient has not been implemented for input with unknown inner matrix shape
#    print ("T shape: ", T.get_shape().as_list())
    D, U, V = tf.svd(T)
    
    corr = tf.reduce_sum(D[:K])
    return -corr    # maximize correlation is to minimze the negative of it

def Inception_V2(input_batch):
    """
    Reference:
        1. Vasili's codes
        2. https://github.com/tensorflow/models/issues/429#issuecomment-277885861
    """

    slim_dir = "/home/xyang/workspace/models/research/slim"
    checkpoints_dir = slim_dir + "/pretrain"
    checkpoints_file = checkpoints_dir + '/inception_v2.ckpt'

    import sys
    sys.path.append(slim_dir)
    from nets import inception
    slim = tf.contrib.slim
    image_size = inception.inception_v2.default_image_size

    cropped_images = tf.random_crop(
           tf.image.convert_image_dtype(input_batch, dtype=tf.float32),
           [tf.shape(input_batch)[0], image_size, image_size, 3])

    preprocessed_images = tf.multiply(tf.subtract(cropped_images, 0.5), 2.0)
        
    # Create the model, use the default arg scope to configure
    # the batch norm parameters.
    with slim.arg_scope(inception.inception_v2_arg_scope()):
        logits, endpoints = inception.inception_v2(preprocessed_images,
                                                        num_classes=1001,
                                                        is_training=True)
        pool5 = endpoints['AvgPool_1a']

    return tf.reshape(pool5, (-1,1024))
