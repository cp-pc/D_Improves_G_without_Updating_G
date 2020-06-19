import numpy as np

class DataLoader():
    '''
    Generator data loading
    '''
    def __init__(self, batch_size, seq_length, padding_token):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.padding_token = padding_token

    def create_batches(self, data_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.padding_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def create_batches_train_list(self, tokens): 
        self.token_stream =  [fill_seq(sentence, padded_length=self.seq_length, fill_token=self.padding_token) for sentence in tokens]
    
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        mask_tmp = ~np.equal(ret, self.padding_token)
        mask = mask_tmp.astype(float)
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret, mask

    def reset_pointer(self):
        self.pointer = 0


class BalanceDisDataloader():
    '''
    Discriminator data loading
    '''
    def __init__(self, batch_size, seq_length, padding_token):
        self.batch_size = batch_size
        self.batch_size_half = int(batch_size / 2)
        self.sentences_po = np.array([])
        self.sentences_neg = np.array([])
        self.sentences = np.array([])
        self.labels = np.array([])
        self.seq_length = seq_length
        self.padding_token = padding_token

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    positive_examples.append(parse_line)
                else:
                    while len(parse_line) < self.seq_length:
                        positive_examples.append(self.padding_token)
                    if len(parse_line) == self.seq_length:
                        positive_examples.append(parse_line)

        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
                else:
                    while len(parse_line) < self.seq_length:
                        negative_examples.append(self.padding_token)
                    if len(parse_line) == self.seq_length:
                        negative_examples.append(parse_line)

        self.sentences_po = np.array(positive_examples)
        self.sentences_neg = np.array(negative_examples)
        # Split batches
        self.num_batch_po = int(len(self.sentences_po) / self.batch_size_half)
        self.sentences_po = self.sentences_po[:self.num_batch_po * self.batch_size_half]
        self.sentences_batches_po = np.split(self.sentences_po, self.num_batch_po, 0)

        self.num_batch_neg = int(len(self.sentences_neg) / self.batch_size_half)
        self.sentences_neg = self.sentences_neg[:self.num_batch_neg * self.batch_size_half]
        self.sentences_batches_neg = np.split(self.sentences_neg, self.num_batch_neg, 0)

        # Generate labels
        positive_labels = [[0, 1] for _ in range(self.batch_size_half)]
        negative_labels = [[1, 0] for _ in range(self.batch_size_half, self.batch_size)]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        self.po_point = 0
        self.neg_point = 0

    def load_train_data_list_file(self, positive_tokens, negative_file):
        # Load data
        positive_examples =  [fill_seq(sentence, padded_length=self.seq_length, fill_token=self.padding_token) for sentence in positive_tokens]
        
        negative_examples = []
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
                else:
                    while len(parse_line) < self.seq_length:
                        negative_examples.append(self.padding_token)
                    if len(parse_line) == self.seq_length:
                        negative_examples.append(parse_line)

        self.sentences_po = np.array(positive_examples)
        self.sentences_neg = np.array(negative_examples)
        # Split batches
        self.num_batch_po = int(len(self.sentences_po) / self.batch_size_half)
        self.sentences_po = self.sentences_po[:self.num_batch_po * self.batch_size_half]
        self.sentences_batches_po = np.split(self.sentences_po, self.num_batch_po, 0)

        self.num_batch_neg = int(len(self.sentences_neg) / self.batch_size_half)
        self.sentences_neg = self.sentences_neg[:self.num_batch_neg * self.batch_size_half]
        self.sentences_batches_neg = np.split(self.sentences_neg, self.num_batch_neg, 0)

        # Generate labels
        positive_labels = [[0, 1] for _ in range(self.batch_size_half)]
        negative_labels = [[1, 0] for _ in range(self.batch_size_half, self.batch_size)]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        self.po_point = 0
        self.neg_point = 0

    def load_train_data_list(self, positive_tokens, negative_tokens):
        # Load data
        positive_examples =  [fill_seq(sentence, padded_length=self.seq_length, fill_token=self.padding_token) for sentence in positive_tokens]
        negative_examples =  [fill_seq(sentence, padded_length=self.seq_length, fill_token=self.padding_token) for sentence in negative_tokens]
    
        # Load data
        #positive_examples = positive_tokens
        #negative_examples = negative_tokens
        
        self.sentences_po = np.array(positive_examples)
        self.sentences_neg = np.array(negative_examples)
        # Split batches
        self.num_batch_po = int(len(self.sentences_po) / self.batch_size_half)
        self.sentences_po = self.sentences_po[:self.num_batch_po * self.batch_size_half]
        self.sentences_batches_po = np.split(self.sentences_po, self.num_batch_po, 0)

        self.num_batch_neg = int(len(self.sentences_neg) / self.batch_size_half)
        self.sentences_neg = self.sentences_neg[:self.num_batch_neg * self.batch_size_half]
        self.sentences_batches_neg = np.split(self.sentences_neg, self.num_batch_neg, 0)

        # Generate labels
        positive_labels = [[0, 1] for _ in range(self.batch_size_half)]
        negative_labels = [[1, 0] for _ in range(self.batch_size_half, self.batch_size)]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        self.po_point = 0
        self.neg_point = 0

    def next_batch(self):
        self.sentences = np.vstack(
            (self.sentences_batches_po[self.po_point], self.sentences_batches_neg[self.neg_point]))

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences_shuffle = self.sentences[shuffle_indices]
        self.labels_shuffle = self.labels[shuffle_indices]

        self.po_point = (self.po_point + 1) % self.num_batch_po
        self.neg_point = (self.neg_point + 1) % self.num_batch_neg

        ret = self.sentences_shuffle, self.labels_shuffle
        return ret

    def random_batch(self):
        rn_pointer_po = np.random.randint(0, self.num_batch_po - 1)
        rn_pointer_neg = np.random.randint(0, self.num_batch_neg - 1)
        self.sentences = np.vstack(
            (self.sentences_batches_po[rn_pointer_po], self.sentences_batches_neg[rn_pointer_neg]))

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences_shuffle = self.sentences[shuffle_indices]
        self.labels_shuffle = self.labels[shuffle_indices]

        ret = self.sentences_shuffle, self.labels_shuffle
        return ret

    def reset_pointer(self):
        self.po_point = 0
        self.neg_point = 0


def fill_seq(input, padded_length, fill_token):
    #padding
    input_padded = input[:]
    input_padded += [int(fill_token)] * (padded_length - len(input))
    return input_padded
