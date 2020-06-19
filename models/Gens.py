from abc import abstractmethod, ABCMeta

from utils.utils import init_sess
from utils.data import *
import os
import numpy as np
import tensorflow as tf
import pickle
from utils.metrics.Nll import Nll


class Gens(metaclass=ABCMeta):

    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.discriminator_d1 = None
        #data load
        self.train_data_loader = None
        self.valid_data_loader = None
        self.test_data_loader = None
        self.fake_data_loader = None
        self.dis_train_data_loader = None
        self.dis_valid_data_loader = None
        # temp file
        self.generator_file_pkl = None
        self.generator_test_file = None
        self.generator_valid_file = None
        self.text_file = None
        # pathes
        self.output_path = None
        self.save_path = None
        self.summary_path = None
        # dict
        self.wi_dict = None
        self.iw_dict = None
        #common
        self.sequence_length = None
        self.vocab_size = None
        self.sess = init_sess()
        self.metrics = list()
        self.log = None
        self.epoch = 0
        self.total_num = 0
        self.keep_num = 0
        self.filt_num = 0
        #generate num
        self.num_generate_train = None
        #train pkl file
        self.train_code = None
        self.valid_code = None
        self.test_code = None


    def set_config(self, config):
        self.__dict__.update(config.dict)


    def set_generator(self, generator=None):
        self.generator = generator


    def set_discriminator(self, discriminator=None, discriminator_d1=None):
        self.discriminator = discriminator
        self.discriminator_d1 = discriminator_d1


    def set_data_loader(self, train_loader, valid_loader, test_loader, fake_loader, dis_train_loader, dis_valid_loader):
        #true sample data loader
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.test_data_loader = test_loader
        #fake sample data loader
        self.fake_data_loader = fake_loader
        #dis data lodaer
        self.dis_train_data_loader = dis_train_loader
        self.dis_valid_data_loader = dis_valid_loader


    def set_sess(self, sess):
        self.sess = sess


    def add_epoch(self):
        self.epoch += 1


    def reset_epoch(self):
        # in use
        self.epoch = 0
        return


    def get_real_test_file(self):
        '''
         Generate Samples test id to word
        :return:
        '''
        with open(self.generator_test_file, 'r') as file:
            codes = get_tokenlized(self.generator_test_file)
        output = id_to_words(ids=codes, idx2word=self.iw_dict)
        with open(self.text_file, 'w', encoding='utf-8') as outfile:
            outfile.write(output)
        output_file = os.path.join(self.output_path, f"epoch_{self.epoch}.txt")
        with open(output_file, 'w', encoding='utf-8') as of:
            of.write(output)


    def generate_samples(self, temperature=1.0):
        '''
        Three samples are generated for discriminator training, verification and testing.
        Stored in three files.
        Samples train num: num_generate_train
        Samples test num: 10000
        Samples valid num: 10000
        :param temperature:When the generator is LSTM, temperature here only works；
                            gpt-2's temperature is set when the model is initialized
        :return:
        '''
        generator = self.generator
        
        # Generate Samples
        generated_samples = []
        for _ in range(int(self.num_generate_train / self.batch_size)+1):
            generated_samples.extend(generator.generate(self.sess, temperature))
        with open(self.generator_file_pkl, 'wb') as out:
            pickle.dump(generated_samples[:self.num_generate_train], out)

        # Generate Samples test
        generated_samples_test = []
        for _ in range(int(10000 / self.batch_size)+1):
            generated_samples_test.extend(generator.generate(self.sess, temperature))
        with open(self.generator_test_file, 'w') as fout:
            for sent in generated_samples_test[:10000]:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)

        # Generate Samples valid
        generated_samples_valid = []
        for _ in range(int(10000 / self.batch_size) + 1):
            generated_samples_valid.extend(generator.generate(self.sess, temperature))
        with open(self.generator_valid_file, 'w') as fout:
            for sent in generated_samples_valid[:10000]:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)

        return generated_samples[:self.num_generate_train], generated_samples_test[:10000], generated_samples_valid[:10000]


    def uc_sampleing(self, generator, discriminator, dn, temperature, c):
        '''
        Adjust uc until c1 is infinitely close to c

        :param generator:
        :param discriminator:
        :param dn:
        :param temperature:When the generator is LSTM, temperature here only works；
                            gpt-2's temperature is set when the model is initialized
        :param c:
        :return:
        '''
        ak = 0
        m = 1000
        #n :Number of cycles
        #uc :uc initial value
        #ds :uc step
        #The above three initial values can be adjusted according to the actual situation
        if c == 0.8:
            n = 10000
            uc = 0.5
            ds = 0.00005
        elif c == 0.5:
            n = 1000
            uc = 0.5
            ds = 0.0005
        else:
            n = 100
            ds = 0.005
            uc = 0.5

        print("up_uc=", 0, " c=", c, " uc=", uc)
        for up_uc in range(n):
            ak = 0
            # Generate Samples
            generated_samples = []
            for _ in range(int(m / self.batch_size)+1):
                generated_samples.extend(generator.generate(self.sess, temperature))
            generated_samples = generated_samples[:m]

            y_batch = [[1, 0] for _ in range(m)]
            random_num = np.random.rand(m)
            feed = {
                discriminator.input_x: generated_samples,
                discriminator.input_y: y_batch,
                discriminator.random_num: random_num,
                discriminator.c: c,
                discriminator.uc: uc
            }
            fake_less_score, fake_samples_score, fake_samples_keep, fake_samples_filterd = self.sess.run([\
            discriminator.fake_less_score,discriminator.fake_samples_score,
            discriminator.fake_samples_keep, discriminator.fake_samples_filterd], feed)
            
            keeps = np.sum(fake_samples_keep, axis=1)
            filters = np.sum(fake_samples_filterd, axis=1)
            keep_num = np.sum(1* [keeps != 0])
            filter_num = np.sum(1* [filters != 0])

            ak = keep_num
            c1 = ak / m
            mark = 1
            if c1 < c:
                mark = -1
            uc = uc + mark * ds
            print("up_uc=", up_uc+1, " c=", c, " c1=", c1, " uc=", uc)
        return uc


    def generate_sample_by_c_uc(self, x_batch, discriminator, dn, c, uc):
        y_batch = [[1, 0] for _ in range(self.batch_size)]
        random_num = np.random.rand(self.batch_size)
        feed = {
            discriminator.input_x: x_batch,
            discriminator.input_y: y_batch,
            discriminator.random_num: random_num,
            discriminator.c: c,
            discriminator.uc: uc
        }
        fake_samples_keep, fake_samples_filterd = self.sess.run([\
        discriminator.fake_samples_keep, discriminator.fake_samples_filterd], feed)

        keeps = np.sum(fake_samples_keep, axis=1)
        filters = np.sum(fake_samples_filterd, axis=1)
        keep_num = np.sum(1* [keeps != 0])
        filterd_num = np.sum(1* [filters != 0])

        if self.keep_num < 10000:
            self.output_samples(fake_samples_keep, f'{dn}_fake_keep_c{int(c*100)}_test')
        elif self.keep_num <= 20000 + self.batch_size:
            self.output_samples(fake_samples_keep, f'{dn}_fake_keep_c{int(c*100)}_valid')
        elif self.keep_num <= self.num_generate_train + 20000 + 2*self.batch_size:
            self.output_train_samples(fake_samples_keep, f'{dn}_fake_keep_c{int(c*100)}_train')
        self.keep_num += keep_num

        if self.filt_num < 10000:
            self.output_samples(fake_samples_filterd, f'{dn}_fake_filter_c{int(c*100)}_test')
        elif self.filt_num <= 20000 + self.batch_size:
            self.output_samples(fake_samples_filterd, f'{dn}_fake_filter_c{int(c*100)}_valid')
        elif self.filt_num <= self.num_generate_train + 20000 + 2*self.batch_size:
            self.output_train_samples(fake_samples_filterd, f'{dn}_fake_filter_c{int(c*100)}_train')
        self.filt_num += filterd_num


    def num_to_0(self):
        print("total_num : "+str(self.total_num*self.batch_size))
        print("keep_num : "+str(self.keep_num))
        print("filt_num : "+str(self.filt_num))
        self.total_num = 0
        self.keep_num = 0
        self.filt_num = 0
        
       
    def output_samples(self, sentences, type):
        with open(os.path.join(self.output_path, f"{type}.txt"), 'a+') as fout:
            for sent in sentences:
                if np.sum(sent) == 0:
                    pass
                else:
                    buffer = ' '.join([str(x) for x in sent]) + '\n'
                    fout.write(buffer)


    def output_train_samples(self, sentences, type):
        with open(os.path.join(self.output_path, f"{type}.pkl"), 'ab+') as out:
            for sent in sentences:
                if np.sum(sent) == 0:
                    pass
                else:
                    pickle.dump(sent.tolist(), out)


    def real_text_samples(self, type):
        '''
        Generate Samples test id to word
        :param type:
        :return:
        '''
        with open(os.path.join(self.output_path, f"{type}_test.txt"), 'r') as file:
            codes = get_tokenlized(os.path.join(self.output_path, f"{type}_test.txt"))
        output = id_to_words(ids=codes, idx2word=self.iw_dict)
        output_file = os.path.join(self.output_path, f"{type}_test_textfile.txt")
        with open(output_file, 'w', encoding='utf-8') as of:
            of.write(output)


    def save_summary(self):
        # summary writer
        self.sum_writer = tf.summary.FileWriter(
            self.summary_path, self.sess.graph)
        return self.sum_writer


    def get_distance(self, fake_file, discriminator, datatype, true_data_loader, epoch, writer):
        '''
        Calculate the accuracy and loss
        :param fake_file:Generate samples
        :param discriminator:
        :param datatype:name
        :param true_data_loader:true samples
        :param epoch:
        :param writer:
        :return:
        '''
        if isinstance(fake_file, list):
            self.fake_data_loader.create_batches_train_list(fake_file)
        else:
            self.fake_data_loader.create_batches(fake_file)

        true_correct_all = []
        true_error_all = []
        true_loss_all = []
        for _ in range(true_data_loader.num_batch):
            x_batch_t, _ = true_data_loader.next_batch()
            y_batch_t = [[0, 1] for _ in range(self.batch_size)]
            feed_t = {
                discriminator.input_x: x_batch_t,
                discriminator.input_y: y_batch_t
            }
            true_loss, true_correct, true_error = self.sess.run(
                [discriminator.loss, discriminator.true_correct,discriminator.true_error],
                feed_t)
            true_correct_all.append(np.sum(true_correct))
            true_error_all.append(np.sum(true_error))
            true_loss_all.append(true_loss)

        fake_correct_all = []
        fake_error_all = []
        fake_loss_all = []
        for _ in range(true_data_loader.num_batch):
            x_batch_f, _ = self.fake_data_loader.next_batch()
            y_batch_f = [[1, 0] for _ in range(self.batch_size)]
            feed = {
                discriminator.input_x: x_batch_f,
                discriminator.input_y: y_batch_f
            }
            fake_loss, fake_correct, fake_error = self.sess.run(
                [discriminator.loss, discriminator.fake_correct, discriminator.fake_error],
                feed)
            fake_correct_all.append(np.sum(fake_correct))
            fake_error_all.append(fake_error)
            fake_loss_all.append(fake_loss)

        loss = (np.sum(true_loss_all) + np.sum(fake_loss_all)) / (true_data_loader.num_batch * self.batch_size * 2)
        accuracy = (np.sum(true_correct_all) + np.sum(fake_correct_all)) / (
                    2 * true_data_loader.num_batch * self.batch_size)
        print(f"accuracy {datatype}:", str(accuracy))
        print(f"loss {datatype}：", str(loss))

        #tensoboard info
        if "valid" in datatype :
            feed = {
                discriminator.valid_acc: accuracy,
                discriminator.valid_loss: loss
            }
            merge_valid = self.sess.run(discriminator.merge_valid,
                feed)
            writer.add_summary(merge_valid, epoch)
        elif "test" in datatype:
            feed = {
                discriminator.test_acc: accuracy,
                discriminator.test_loss: loss
                }
            merge_test = self.sess.run(discriminator.merge_test,
                feed)
            writer.add_summary(merge_test, epoch)

        return accuracy, loss


    def get_fake_code(self, pkl_path):
        #read pkl file
        #The filtered sample is stored sentence by sentence, so it needs to be read cyclically
        fake_codes = []
        with open(pkl_path, 'rb') as f:
            while True:
                try:
                    if len(fake_codes) >= self.num_generate_train:
                        break
                    fake_code = pickle.load(f)
                    fake_codes.append(fake_code)
                except EOFError:
                    break
        return fake_codes


    def lm_scores(self, generator, fake_data_loader, sess):
        lm = Nll(fake_data_loader, generator, sess)
        lm_score = lm.get_score()
        print("lm_score:", lm_score)
        return lm_score


    def train_real(self):
        pass


class Gen(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def run_epoch(self):
        pass


class Dis(metaclass=ABCMeta):

    def __init__(self):
        pass

