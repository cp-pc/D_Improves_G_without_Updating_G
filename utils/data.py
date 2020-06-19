import os
import re
import _pickle as pickle

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_set = set()
        self.max_seq_len = 0

        # add <unk> <sos> and <eos> tokens
        # really important not to change (hardcoded in minibatch_generator)
        self.add_word(u'<pad>')  # ID 0
        self.add_word(u'<eos>')  # ID 1
        self.add_word(u'<sos>')  # ID 2
        self.add_word(u'<unk>')  # ID 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __seq_len__(self):
        return self.max_seq_len

def tokenize(data_file, data_ids_file, vocab_file, train=False, word_dict=None, char_level=False, dataset=None, skip=False):
    # tokenizing process is somewhat lenghty. Let's try to avoid 
    # it when possible
    if not skip:
        try:
            #path_word_dict = path + '_word_dict.pickle'
            #path_ids = path + '_ids.pickle'
            path_word_dict = vocab_file
            path_ids = data_ids_file
            with open(path_ids, 'rb') as f: 
                ids = pickle.load(f)
            if train: 
                with open(path_word_dict, 'rb') as f: 
                    word_dict = pickle.load(f)
            
            print('loaded preprocessed data from %s' % data_file)
            return ids, word_dict
        except: 
            pass

    """Tokenizes a text file."""
    if word_dict is None : 
        print('creating new word dictionary')
        word_dict = Dictionary() 
    assert os.path.exists(data_file), '{} does not exist'.format(data_file)
    # Add words to the dictionary
    with open(data_file, 'r', encoding='utf-8') as f:
        sentences = 0
        max_tokens = 0
        for line in f:
            # line = line.decode('utf-8', 'strict')
            words = re.findall(r"[\w']+|[.,!?;]", line,
                    flags=re.UNICODE) 
            
            if char_level: 
                chars = []
                for word in words: 
                    for cc in word: 
                        chars += [cc]
                    chars += [' ']
            
                # remove last space
                chars = chars[:-1]
                words = chars
            else: 
                if words[-1] == '.':
                    words[-1] = '<eos>'
                elif words[-1] == '?':
                    words[-1] =  '<qm>'
                elif words[-1] == '!':
                    words[-1]  ='<em>'
            
            # track stats for building tokenized version
            tokens = len(words)
            sentences += 1
            if tokens > max_tokens:
                max_tokens = tokens
            
            # only add words if in training set
            if train:
                for word in words:
                    word_dict.add_word(word)
                word_dict.vocab_set = \
                    set(word_dict.idx2word)
                
                word_dict.max_seq_len = max_tokens

    # Tokenize file content
    with open(data_file, 'r',encoding='utf-8') as f:
        ids = []
        for i, line in enumerate(f):
            #line = line.decode('utf-8', 'strict')
            words = re.findall(r"[\w']+|[.,!?;]", line, 
                    flags=re.UNICODE)
            
            if char_level: 
                chars = []
                for word in words: 
                    for cc in word: 
                        chars += [cc]
                    chars += [' ']
            
                # remove last space
                chars = chars[:-1]
                words = chars
            else: 
                if words[-1] == '.':
                    words[-1] = '<eos>'
                elif words[-1] == '?':
                    words[-1] =  '<qm>'
                elif words[-1] == '!':
                    words[-1]  ='<em>'

            token = 0
            idx = list(range(len(words)))
            for word in words:
                if word not in word_dict.vocab_set:
                    word = u'<unk>'
                idx[token] = word_dict.word2idx[word]
                token += 1

            # create list of lists for easier process later on
            ids.append(idx)

    # save to file 
    path_word_dict = vocab_file
    path_ids = data_ids_file
    with open(path_ids, 'wb') as f: 
        pickle.dump(ids, f)
    if train: 
        with open(path_word_dict, 'wb') as f: 
            pickle.dump(word_dict, f)
    
    return ids, word_dict


def id_to_words(ids, idx2word):
    paras = ''
    sentences = []
    for sentence in ids:
        human_readable = []
        for word in sentence:
            human_readable.append(idx2word[word])
        human_readable = ' '.join(human_readable)
        human_readable = human_readable.replace('<eos>', '.').replace(' <pad>', '')
        human_readable = human_readable.replace('<qm>', '?').replace('<em>', '!')

        sent_str = str(human_readable)  # [1:-1]
        sent_str = sent_str.replace('\n', ' ')
        paras += sent_str+'\n'
        #sentences.append(human_readable)
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file, 'r') as raw:
        for line in raw:
            line = line.strip().split()
            parse_line = [int(x) for x in line]
            tokenlized.append(parse_line)
    return tokenlized
