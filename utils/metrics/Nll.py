from utils.metrics.Metrics import Metrics


class Nll(Metrics):
    def __init__(self, data_loader, rnn, sess):
        super().__init__()
        self.name = 'nll-oracle'
        self.data_loader = data_loader
        self.sess = sess
        self.rnn = rnn

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.nll_loss()

    def nll_loss(self):
        nll = self.rnn.eval_epoch(self.sess, self.data_loader) #return nll
        return nll
