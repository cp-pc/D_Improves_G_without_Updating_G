import numpy as np

from utils.metrics.Metrics import Metrics


class PPL(Metrics):
    def __init__(self, data_loader, rnn, sess):
        super().__init__()
        self.name = 'ppl-oracle'
        self.data_loader = data_loader
        self.sess = sess
        self.rnn = rnn

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.ppl_loss()

    def ppl_loss(self):
        nll = self.rnn.eval_epoch(self.sess, self.data_loader) #return nll
        return np.exp(nll)