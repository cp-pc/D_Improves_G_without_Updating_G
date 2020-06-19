import json
from time import time

from models.Gens import Gens
from models.gpt2.DataLoader import DataLoader,BalanceDisDataloader
from models.gpt2.Discriminator import Discriminator
from models.gpt2.GPT2 import Generator
from utils.utils import *
from colorama import Fore
import tensorflow.contrib.slim as slim
import pickle
from utils.data import *
from utils.metrics.Nll import Nll
from utils.metrics.Bleu import Bleu
from utils.metrics.SelfBleu import SelfBleu
from utils.metrics.FED import FED

class GPT2D(Gens):
    def __init__(self):
        super().__init__()

    def train_discriminator(self, discriminator):
        '''
        Discriminator training
        :param discriminator
        :return:
        '''
        for step in range(self.dis_train_data_loader.num_batch_po):
            x_batch, y_batch = self.dis_train_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch
            }
            self.sess.run(discriminator.train_op, feed)

    def init_real_trainng(self):
        '''
        Build and initialize the model
        :return:
        '''
        initializer = tf.random_uniform_initializer(-0.05, 0.05)

        with tf.variable_scope("generator_gpt2", reuse=None, initializer=initializer):
            generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size,  sequence_length=self.sequence_length, \
                                  generator_name="generator_gpt2", start_token=self.start_token, \
                                  learning_rate=self.gen_lr, temperature=self.temperature, n_embd=self.n_embd,\
                                  n_head=self.n_head, n_layer=self.n_layer, dropout_keep_prob=self.g_dropout_keep_prob)

        self.set_generator(generator=generator)

        with tf.variable_scope("discriminator_base"):
            discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                          vocab_size=self.vocab_size,
                                          discriminator_name='discriminator_base', emd_dim=self.emb_dim,
                                          filter_sizes=self.filter_size, num_filters=self.num_filters,
                                          l2_reg_lambda=self.l2_reg_lambda, dis_lr=self.dis_lr)

        with tf.variable_scope("discriminator_d1"):
            discriminator_d1 = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                             vocab_size=self.vocab_size,
                                             discriminator_name='discriminator_d1', emd_dim=self.emb_dim,
                                             filter_sizes=self.filter_size, num_filters=self.num_filters,
                                             l2_reg_lambda=self.l2_reg_lambda, dis_lr=self.dis_lr)

        self.set_discriminator(discriminator=discriminator, discriminator_d1=discriminator_d1)

        # dataloder
        train_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        valid_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        test_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        fake_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        # discriminator　dataloder
        dis_train_dataloader = BalanceDisDataloader(batch_size=self.dis_batch_size, seq_length=self.sequence_length,
                                                    padding_token=0)
        dis_valid_dataloader = BalanceDisDataloader(batch_size=self.dis_batch_size, seq_length=self.sequence_length,
                                                    padding_token=0)

        self.set_data_loader(train_loader=train_dataloader, valid_loader=valid_dataloader, test_loader=test_dataloader,
                             fake_loader=fake_dataloader, dis_train_loader=dis_train_dataloader,
                             dis_valid_loader=dis_valid_dataloader)
        
    def train(self):
        self.init_real_trainng()
        ###
        #init_metric:NLL
        ###
        gen_valid_nll = Nll(self.valid_data_loader, self.generator, self.sess)

        FLAGS = tf.app.flags.FLAGS
        if FLAGS.data == 'image_coco':
            self.valid_code = self.test_code
        self.valid_data_loader.create_batches_train_list(self.valid_code)
        self.train_data_loader.create_batches_train_list(self.train_code)
        self.test_data_loader.create_batches_train_list(self.test_code)
        self.sess.run(tf.global_variables_initializer())

        # ++ Saver
        # saver_variables = tf.global_variables
        saver_variables = slim.get_variables_to_restore(include=["generator_gpt2"])
        saver = tf.train.Saver(saver_variables, max_to_keep=20)
        # ++ ====================
        
        # summary writer
        self.writer = self.save_summary()

        if self.restore:
            restore_from = tf.train.latest_checkpoint(os.path.join(self.save_path, "gen"))
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
        else:
            best_nll = 1000
            print('start train generator:')
            for epoch in range(self.train_gen_num):
                start = time()
                loss = self.generator.run_epoch(self.sess, self.train_data_loader, self.writer, epoch)
                end = time()
                print('epoch:' + str(epoch) + ' loss: ' + str(loss) + ' \t time:' + str(end - start))
                if (epoch + 1) % self.ntest == 0:
                    values = gen_valid_nll.get_score()
                    if values < best_nll:
                        best_nll = values
                        # save pre_train
                        saver.save(self.sess, os.path.join(self.save_path, "gen", 'train_best'))
                        print('gen store')
                self.add_epoch()
            restore_from = tf.train.latest_checkpoint(os.path.join(self.save_path, "gen"))
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")


            print('start train discriminator:')
            saver_variables = slim.get_variables_to_restore(include=["discriminator_base"])
            saver = tf.train.Saver(saver_variables, max_to_keep=20)

            self.generate_samples(self.temperature)
            self.get_real_test_file()
            with open(self.generator_file_pkl, 'rb')  as inf:
                self.generator_code = pickle.load(inf)
            self.dis_train_data_loader.load_train_data_list(self.train_code, self.generator_code)
            self.dis_valid_data_loader.load_train_data_list_file(self.valid_code, self.generator_valid_file)
            acc_valid_best = 0
            for epoch in range(self.train_dis_num):
                print("base epoch:" + str(epoch))
                self.train_discriminator(self.discriminator)

                accuracy_valid, loss_valid = self.get_distance(self.generator_valid_file, self.discriminator, \
                "base_valid", self.valid_data_loader, epoch, self.writer)
                if accuracy_valid > acc_valid_best:
                        acc_valid_best = accuracy_valid
                        saver.save(self.sess, os.path.join(self.save_path, f'train_base'))
            print("acc_valid_best base:", acc_valid_best)

            restore_from = os.path.join(self.save_path, "train_base")
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
            accuracy_test, loss_test = self.get_distance(self.generator_test_file, self.discriminator, \
                                                                  "base_test", self.test_data_loader, epoch,
                                                                  self.writer)
            print("acc_test base:", accuracy_test)

            #####
            # Filter generation
            #####
            for c_give in [0.2, 0.5, 0.8]:
                print("*" * 50)
                print("c", c_give)
                # c_give is the acceptance ratio
                uc_get = self.uc_sampleing(self.generator, self.discriminator, "d1", self.temperature, c=c_give)
                print("uc=", uc_get)

                self.total_num = 0
                while self.keep_num < self.num_generate_train + 20000 + self.batch_size:
                    inp = self.generator.generate(self.sess, self.temperature)
                    self.generate_sample_by_c_uc(inp, self.discriminator, 'd1', c_give, uc_get)
                    self.total_num += 1
                self.num_to_0()
                self.real_text_samples(f'd1_fake_keep_c{int(c_give * 100)}')
                self.real_text_samples(f'd1_fake_filter_c{int(c_give * 100)}')
                print("==" * 50)

            #####
            # Test the accuracy of filtered samples
            #####
            print(f'start train discriminator_d1 ')
            filname = 20
            saver_variables = slim.get_variables_to_restore(include=["discriminator_d1"])
            saver = tf.train.Saver(saver_variables, max_to_keep=20)

            self.d1_fake_codes = self.get_fake_code(
                os.path.join(self.output_path, f"d1_fake_keep_c{filname}_train.pkl"))
            self.dis_train_data_loader.load_train_data_list(self.train_code, self.d1_fake_codes)
            self.dis_valid_data_loader.load_train_data_list_file(self.valid_code, os.path.join(self.output_path,
                                                                                               f"d1_fake_keep_c{filname}_valid.txt"))
            acc_valid_best = 0
            for epoch in range(self.train_dis_num):
                print("d1 epoch:" + str(epoch))
                self.train_discriminator(self.discriminator_d1)

                accuracy_valid, loss_valid = self.get_distance(
                    os.path.join(self.output_path, f"d1_fake_keep_c{filname}_valid.txt"), self.discriminator_d1, \
                    "d1_valid", self.valid_data_loader, epoch, self.writer)
                if accuracy_valid > acc_valid_best:
                    acc_valid_best = accuracy_valid
                    saver.save(self.sess, os.path.join(self.save_path, f'train_d1'))
            print("acc_valid_best d1:", acc_valid_best)
            restore_from = os.path.join(self.save_path, "train_d1")
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
            accuracy_test, loss_test = self.get_distance(
                os.path.join(self.output_path, f"d1_fake_keep_c{filname}_test.txt"), self.discriminator_d1, \
                "d1_test", self.test_data_loader, epoch, self.writer)
            print(f"c={filname / 100}", "acc_test d1:", accuracy_test)

    ###
    #Evaluation: lm_vs_rlm, bleu_vs_sbleu, fed
    ###

    def rlm_scores(self, re_generator, fake_data_loader, sess, test_data_loader, valid_data_loader, writer):
        test_rlm = Nll(test_data_loader, re_generator, sess)
        valid_rlm = Nll(valid_data_loader, re_generator, sess)
        print('start train re-generator:')
        valid_rlm_best = 1000
        test_rlm_best = 0
        self.re_gen_num = 80
        for epoch in range(self.re_gen_num):
            start = time()
            loss = re_generator.run_epoch(sess, fake_data_loader, writer, epoch)
            end = time()
            print('epoch:' + str(epoch) + ' loss: ' + str(loss) + ' \t time:' + str(end - start))

            test_rlm_score = test_rlm.get_score()
            valid_rlm_score = valid_rlm.get_score()
            print('valid_rlm_score:' + str(valid_rlm_score) + '   test_rlm_score: ' + str(test_rlm_score))
            if (epoch + 1) % self.ntest == 0:
                if valid_rlm_score < valid_rlm_best:
                    valid_rlm_best = valid_rlm_score
                    test_rlm_best = test_rlm_score
        print('valid_rlm_best:' + str(valid_rlm_best) + 'test_rlm_best: ' + str(test_rlm_best))

        return valid_rlm_best, test_rlm_best

    def lm_vs_rlm(self, model_path, fake_train_code_file, fake_test_code_file):
        '''
        Run the train_real method and get the model, fake_train_code, fake_test_code
        :param model_path:
        :param fake_train_code_file: .pkl file
        :param fake_test_code_file: ___test.txt file
        :return:
        '''

        initializer = tf.random_uniform_initializer(-0.05, 0.05)

        with tf.variable_scope("generator_gpt2", reuse=None, initializer=initializer):
            generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size,  sequence_length=self.sequence_length, \
                                  generator_name="generator_gpt2", start_token=self.start_token, \
                                  learning_rate=self.gen_lr, temperature=self.temperature, n_embd=self.n_embd,\
                                  n_head=self.n_head, n_layer=self.n_layer, dropout_keep_prob=self.g_dropout_keep_prob)

        self.set_generator(generator=generator)


        # dataloder
        train_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        valid_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        test_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)
        fake_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length, padding_token=0)

        self.set_data_loader(train_loader=train_dataloader, valid_loader=valid_dataloader, test_loader=test_dataloader,
                             fake_loader=fake_dataloader, dis_train_loader=None,
                             dis_valid_loader=None)

        FLAGS = tf.app.flags.FLAGS
        if FLAGS.data == 'image_coco':
            self.valid_code = self.test_code
        self.valid_data_loader.create_batches_train_list(self.valid_code)
        self.train_data_loader.create_batches_train_list(self.train_code)
        self.test_data_loader.create_batches_train_list(self.test_code)
        self.sess.run(tf.global_variables_initializer())

        # ++ Saver
        # saver_variables = tf.global_variables
        saver_variables = slim.get_variables_to_restore(include=["generator_gpt2"])
        saver = tf.train.Saver(saver_variables, max_to_keep=20)
        # ++ ====================

        # summary writer
        self.writer = self.save_summary()

        print("-- gpt2 lm_vs_rlm--")
        saver.restore(self.sess, model_path)
        print(f"{Fore.BLUE}Restore from : {model_path}{Fore.RESET}")

        # load fake test code file(txt)
        self.fake_data_loader.create_batches(fake_test_code_file)
        self.lm_scores(self.generator, self.fake_data_loader, self.sess)

        # load fake train code file(pkl)
        if "tmp" in fake_train_code_file:
            with open(fake_train_code_file, 'rb') as inf:
                fake_train_code = pickle.load(inf)  # The unfiltered samples are stored once, so they are read once
        else:
            fake_train_code = self.get_fake_code(
                fake_train_code_file)  # The filtered sample is stored sentence by sentence, so it needs to be read cyclically
        self.fake_data_loader.create_batches_train_list(fake_train_code)

        with tf.variable_scope("re_generator", reuse=None, initializer=initializer):
            generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size,  sequence_length=self.sequence_length, \
                                  generator_name="re_generator", start_token=self.start_token, \
                                  learning_rate=self.gen_lr, temperature=self.temperature, n_embd=self.n_embd,\
                                  n_head=self.n_head, n_layer=self.n_layer, dropout_keep_prob=self.g_dropout_keep_prob)

        self.set_generator(generator=generator)
        self.sess.run(tf.global_variables_initializer())

        self.rlm_scores(self.generator, self.fake_data_loader, self.sess, self.test_data_loader, self.valid_data_loader,
                        self.writer)

    def bleu_vs_sbleu(self, real_text, fake_text):
        bleu = Bleu(test_text=fake_text, real_text=real_text, gram=5)
        sbleu = SelfBleu(test_text=fake_text, gram=5)
        print("Bleu:", bleu.get_score(), "SelfBleu", sbleu.get_score())

    def fed(self, real_text, fake_text):
        '''
        require Tensorflow 2.0
        and hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
        :param real_text:
        :param fake_text:
        :return:
        '''
        fed = FED(test_text=fake_text, real_text=real_text)
        print("FED:", fed.get_score())