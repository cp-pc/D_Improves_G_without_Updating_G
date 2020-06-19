import tensorflow as tf
import numpy as np
from models.Gens import Gen

class Generator(Gen):
    def __init__(self, num_vocabulary, batch_size, hidden_dim, sequence_length, generator_name,
                 learning_rate=0.001, grad_clip=10.0, dropout_keep_prob=0.5,\
                start_token=2, num_layers_gen=2):
        self.num_vocabulary = num_vocabulary
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.num_layers_gen = num_layers_gen
        self.run_keep_drop = dropout_keep_prob
        self.temperature = tf.placeholder(tf.float32)

        # tensor placeholder
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(tf.int32, [self.batch_size, None])
            self.targets = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
            self.input_mask = tf.placeholder(tf.float32, [self.batch_size, self.sequence_length])
            self.dropout_keep_place = tf.placeholder(tf.float32)

            self.embedding = tf.get_variable("embedding", [self.num_vocabulary, self.hidden_dim])
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
            inputs = tf.nn.dropout(inputs, self.dropout_keep_place)

        def get_cell(hidden_dim, keep_prob):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

        with tf.variable_scope("rnns"):
            cells = tf.contrib.rnn.MultiRNNCell([get_cell(self.hidden_dim, self.dropout_keep_place) for _ in range(self.num_layers_gen)],
                                                state_is_tuple=True)
            self.initial_state = cells.zero_state(self.batch_size, tf.float32)  
            self.initial_state_gen_x = cells.zero_state(self.batch_size, tf.float32)

        outputs = []
        state = self.initial_state
        state_gen_x = self.initial_state_gen_x
        gen_x = []
        self.gen_x_batch = []

        weight = tf.get_variable("weight", [self.hidden_dim, self.num_vocabulary])
        bias = tf.get_variable("bias", [self.num_vocabulary])
        
        with tf.variable_scope("generator"):
            for time_step in range(self.sequence_length):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cells(inputs[:, time_step, :], state)
                outputs.append(cell_output)

            for t in range(self.sequence_length):
                tf.get_variable_scope().reuse_variables()
                if t > 0:
                    gen_in = tf.nn.embedding_lookup(self.embedding, gen_x_tmp)
                else:
                    gen_in = inputs
                cell_output_gen_x, state_gen_x = cells(gen_in[:, 0, :], state_gen_x)
                logit_step = tf.matmul(cell_output_gen_x, weight) + bias
                logit_step = logit_step * self.temperature #logit_step: [batch, vocab]
                gen_x_tmp = tf.cast(tf.multinomial(logit_step, 1), tf.int32) 
                gen_x.append(gen_x_tmp)
            self.gen_x_batch = tf.concat(gen_x, 1)

        output = tf.reshape(tf.concat(outputs, 1), [-1, self.hidden_dim])
        logits = tf.matmul(output, weight) + bias 

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.batch_size * self.sequence_length], dtype=tf.float32)])

        self.cost = tf.reduce_sum(loss) / (self.sequence_length*self.batch_size)
        self.eval_cost = tf.reduce_sum(loss * tf.reshape(self.input_mask, [-1])) / tf.reduce_sum(self.input_mask) 
        self.final_state = state

        self.params = [param for param in tf.trainable_variables() if generator_name in param.name]

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.params), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, self.params))

        g_train_loss =  tf.summary.scalar('g_train_loss', self.cost)
        self.merge_summary_train = tf.summary.merge([g_train_loss])

    def run_epoch(self, sess, data_loader, writer, epoch):
        total_costs = 0.0
        data_loader.reset_pointer()

        for it in range(data_loader.num_batch):
            y, mask = data_loader.next_batch()
            
            x = np.ones((self.batch_size, self.sequence_length), dtype=int)
            x[:, 0] = self.start_token
            x[:, 1:] = y[:, :self.sequence_length-1]

            cost, _, _, merge_summary_train = sess.run([ self.cost, self.final_state, self.train_op,self.merge_summary_train],
                                        {self.input_data: x, self.targets: y, self.input_mask:mask, self.dropout_keep_place:self.run_keep_drop})
            
            writer.add_summary(merge_summary_train, epoch*data_loader.num_batch+it)
            total_costs += cost

        nll_avr = total_costs/data_loader.num_batch
        return nll_avr

        
    def eval_epoch(self, sess, data_loader):
        total_costs = 0.0
        data_loader.reset_pointer()

        for it in range(data_loader.num_batch):
            y, mask = data_loader.next_batch()
            x = np.ones((self.batch_size, self.sequence_length), dtype=int)
            x[:, 0] = self.start_token
            x[:, 1:] = y[:, :self.sequence_length-1]

            eval_cost, _= sess.run([self.eval_cost, self.final_state],
                                        {self.input_data: x, self.targets: y, self.input_mask:mask, self.dropout_keep_place:1.0})
                        
            total_costs += eval_cost

        nll_avr = total_costs/data_loader.num_batch
        return nll_avr

    def generate(self, sess, temperature=1.0):
        input = np.ones((self.batch_size, 1), dtype=int)*2
        outputs = sess.run(self.gen_x_batch, {self.input_data:input,
                                              self.dropout_keep_place:1.0,
                                              self.temperature:temperature})
        return outputs.tolist()

