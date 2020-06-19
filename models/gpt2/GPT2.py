import tensorflow as tf
import numpy as np
from models.Gens import Gen
from tensorflow.contrib.training import HParams

class Generator(Gen):
    def __init__(self, num_vocabulary, batch_size, sequence_length, generator_name, start_token=2, learning_rate=2e-5,
                 temperature=1.0, n_embd=768, n_head=12, n_layer=12, dropout_keep_prob=0.9):
        self.num_vocabulary = num_vocabulary
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.start_token = start_token
        self.learning_rate = learning_rate
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout_keep_prob = dropout_keep_prob
        self.generator_name = generator_name
        self.temperature = temperature
        self.input = tf.placeholder(tf.int32, [self.batch_size, None])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, None])
        hparam = self.default_hparams()

        #train
        output = self.model(hparams=hparam, X=self.input, scope=self.generator_name, train=True)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.targets, logits=output['logits']))
        train_vars = [param for param in tf.trainable_variables() if self.generator_name in param.name]
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt_grads = tf.gradients(self.loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        self.opt_apply = opt.apply_gradients(opt_grads)

        g_train_loss = tf.summary.scalar('g_train_loss', self.loss)
        self.merge_summary_train = tf.summary.merge([g_train_loss])
        #generator
        self.tf_sample = self.sample_sequence()
        #eval
        eval_output = self.model(hparams=hparam, X=self.input, scope=self.generator_name)
        self.score = tf.nn.softmax(eval_output['logits'])
        self.eval_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.targets, logits=eval_output['logits'])

    def dropout(self, x, train):
        if train and self.dropout_keep_prob > 0:
            x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)
        return x

    def default_hparams(self):
        return HParams(
            n_vocab=self.num_vocabulary,
            n_ctx=self.sequence_length,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
            n_classes=2
        )

    def shape_list(self, x):
        """Deal with dynamic shape in tensorflow cleanly."""
        static = x.shape.as_list()
        dynamic = tf.shape(x)
        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    def softmax(self, x, axis=-1):
        x = x - tf.reduce_max(x, axis=axis, keepdims=True)
        ex = tf.exp(x)
        return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

    def gelu(self, x):
        return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

    def norm(self, x, scope, *, axis=-1, epsilon=1e-5):
        """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
        with tf.variable_scope(scope):
            n_state = x.shape[-1].value
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
            u = tf.reduce_mean(x, axis=axis, keepdims=True)
            s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
            x = (x - u) * tf.rsqrt(s + epsilon)
            x = x * g + b
            return x

    def split_states(self, x, n):
        """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
        *start, m = self.shape_list(x)
        return tf.reshape(x, start + [n, m // n])

    def merge_states(self, x):
        """Smash the last two dimensions of x into a single dimension."""
        *start, a, b = self.shape_list(x)
        return tf.reshape(x, start + [a * b])

    def conv1d(self, x, scope, nf, *, w_init_stdev=0.02):
        with tf.variable_scope(scope):
            *start, nx = self.shape_list(x)
            w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, start + [nf])
            return c

    def attention_mask(self, nd, ns, *, dtype):
        """1's in the lower triangle, counting from the lower right corner.

        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def attn(self, x, scope, n_state, *, past, hparams, train=False):
        assert x.shape.ndims == 3  # Should be [batch, sequence, features]
        assert n_state % hparams.n_head == 0
        if past is not None:
            assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

        def split_heads(x):
            # From [batch, sequence, features] to [batch, heads, sequence, features]
            return tf.transpose(self.split_states(x, hparams.n_head), [0, 2, 1, 3])

        def merge_heads(x):
            # Reverse of split_heads
            return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

        def mask_attn_weights(w):
            # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
            _, _, nd, ns = self.shape_list(w)
            b = self.attention_mask(nd, ns, dtype=w.dtype)
            b = tf.reshape(b, [1, 1, nd, ns])
            w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
            return w

        def multihead_attn(q, k, v):
            # q, k, v have shape [batch, heads, sequence, features]
            w = tf.matmul(q, k, transpose_b=True)
            w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

            w = mask_attn_weights(w)
            w = self.softmax(w)
            
            w = self.dropout(w, train)

            a = tf.matmul(w, v)
            return a

        with tf.variable_scope(scope):
            c = self.conv1d(x, 'c_attn', n_state * 3)
            q, k, v = map(split_heads, tf.split(c, 3, axis=2))
            present = tf.stack([k, v], axis=1)
            if past is not None:
                pk, pv = tf.unstack(past, axis=1)
                k = tf.concat([pk, k], axis=-2)
                v = tf.concat([pv, v], axis=-2)
            a = multihead_attn(q, k, v)
            a = merge_heads(a)
            a = self.conv1d(a, 'c_proj', n_state)
            a = self.dropout(a, train)
            return a, present

    def mlp(self, x, scope, n_state, *, hparams, train=False):
        with tf.variable_scope(scope):
            nx = x.shape[-1].value
            h = self.gelu(self.conv1d(x, 'c_fc', n_state))
            h2 = self.conv1d(h, 'c_proj', nx)
            h2 = self.dropout(h2, train)
            return h2

    def block(self, x, scope, *, past, hparams, train=False):
        with tf.variable_scope(scope):
            nx = x.shape[-1].value
            a, present = self.attn(self.norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams, train=train)
            x = x + a
            m = self.mlp(self.norm(x, 'ln_2'), 'mlp', nx * 4, hparams=hparams, train=train)
            x = x + m
            return x, present

    def past_shape(self, *, hparams, batch_size=None, sequence=None):
        return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

    def expand_tile(self, value, size):
        """Add a new axis of given size."""
        value = tf.convert_to_tensor(value, name='value')
        ndims = value.shape.ndims
        return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)

    def positions_for(self, tokens, past_length):
        batch_size = tf.shape(tokens)[0]
        nsteps = tf.shape(tokens)[1]
        return self.expand_tile(past_length + tf.range(nsteps), batch_size)

    def model(self, hparams, X, scope, past=None, train=False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            results = {}
            batch, sequence = self.shape_list(X)

            wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                                  initializer=tf.random_normal_initializer(stddev=0.01))
            self.wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
            past_length = 0 if past is None else tf.shape(past)[-2]
                
            wpe = self.dropout(wpe, train)
            wte = self.dropout(self.wte, train)
            h = tf.gather(wte, X) + tf.gather(wpe, self.positions_for(X, past_length))

            # Transformer
            presents = []
            pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
            assert len(pasts) == hparams.n_layer
            for layer, past in enumerate(pasts):
                h, present = self.block(h, 'h%d' % layer, past=past, hparams=hparams, train=train)
                if layer == 10:
                    tf.add_to_collection('checkpoints', h)
                presents.append(present)
            results['present'] = tf.stack(presents, axis=1)
            h = self.norm(h, 'ln_f')

            # Language model loss.  Do tokens <n predict token n?
            h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
            logits = tf.matmul(h_flat, wte, transpose_b=True)
            logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
            results['logits'] = logits

            return results

    def sample_sequence(self):
        hparams = self.default_hparams()
        context = tf.fill([self.batch_size, 1], self.start_token)

        def step(hparams, tokens, past=None):
            lm_output = self.model(hparams=hparams, X=tokens, scope=self.generator_name, past=past, reuse=tf.AUTO_REUSE)

            logits = lm_output['logits'][:, :, :hparams.n_vocab]
            presents = lm_output['present']
            presents.set_shape(self.past_shape(hparams=hparams, batch_size=self.batch_size))
            return {
                'logits': logits,
                'presents': presents,
            }

        with tf.name_scope('sample_sequence'):
            # Don't feed the last context token -- leave that to the loop below
            # TODO: Would be slightly faster if we called step on the entire context,
            # rather than leaving the last token transformer calculation to the while loop.
            context_output = step(hparams, context[:, :-1])

            def body(past, prev, output):
                next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
                logits = next_outputs['logits'][:, -1, :] * tf.to_float(self.temperature)
                samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
                return [
                    tf.concat([past, next_outputs['presents']], axis=-2),
                    tf.squeeze(samples, axis=[1]),
                    tf.concat([output, samples], axis=1),
                ]

            def cond(*args):
                return True

            _, _, tokens = tf.while_loop(
                cond=cond, body=body,
                maximum_iterations=self.sequence_length,
                loop_vars=[
                    context_output['presents'],
                    context[:, -1],
                    context,
                ],
                shape_invariants=[
                    tf.TensorShape(self.past_shape(hparams=hparams, batch_size=self.batch_size)),
                    tf.TensorShape([self.batch_size]),
                    tf.TensorShape([self.batch_size, None]),
                ],
                back_prop=False,
            )

            return tokens

    def run_epoch(self, sess, data_loader, writer, epoch):
        total_costs = 0.0
        data_loader.reset_pointer()

        for it in range(data_loader.num_batch):

            y, mask = data_loader.next_batch()

            x = np.ones((self.batch_size, self.sequence_length), dtype=int)
            x[:, 0] = self.start_token
            x[:, 1:] = y[:, :self.sequence_length - 1]

            cost, _, merge_summary_train = sess.run([self.loss, self.opt_apply, self.merge_summary_train],
                                                    {self.input: x, self.targets: y})

            writer.add_summary(merge_summary_train, epoch * data_loader.num_batch + it)
            total_costs += cost

        nll_avr = total_costs / data_loader.num_batch
        return nll_avr

    def eval_epoch(self, sess, data_loader):
        total_costs = 0.0
        data_loader.reset_pointer()

        for it in range(data_loader.num_batch):
            y, mask = data_loader.next_batch()
            x = np.ones((self.batch_size, self.sequence_length), dtype=int)
            x[:, 0] = self.start_token
            x[:, 1:] = y[:, :self.sequence_length - 1]

            costs = sess.run(self.eval_losses, {self.input: x, self.targets: y})

            total_costs += np.sum(costs * mask) / np.sum(mask)

        nll_avr = total_costs / data_loader.num_batch
        return nll_avr

    def generate(self, sess, temperature=1.0):
        #temperature here is useless
        outputs = sess.run(self.tf_sample)
        return outputs[:, 1:].tolist()
