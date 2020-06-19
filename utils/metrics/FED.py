from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_hub as hub
import numpy as np
from scipy import linalg

from utils.metrics.Metrics import Metrics

#**********************
#Tensorflow 2.0 required
#***********************
class FED(Metrics):
    def __init__(self, test_text='', real_text='', num_real_sentences=10000, num_fake_sentences=10000):
        super().__init__()
        self.name = 'FED'
        self.test_data = test_text
        self.real_data = real_text
        self.num_real_sentences = num_real_sentences
        self.num_fake_sentences = num_fake_sentences
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")


    def get_name(self):
        return self.name

    def get_score(self):
        embed_r = self.embed(self.real_data[:self.num_real_sentences])["outputs"].numpy()
        embed_g = self.embed(self.test_data[:self.num_fake_sentences])["outputs"].numpy()
        score = self.fed_score(embed_r, embed_g)
        return score
        
    def fed_score(self, embed_r, embed_g):
        Real_np = embed_r
        Fake_np = embed_g

        U_real = np.mean(Real_np, axis=0)
        U_fake = np.mean(Fake_np, axis=0)
        C_real = np.cov(Real_np.transpose())
        C_fake = np.cov(Fake_np.transpose())

        U_x_y = U_real.dot(U_real) + U_fake.dot(U_fake) - 2 * U_fake.dot(U_real)
        C_r_f_sqrt = linalg.sqrtm(C_real.dot(C_fake), True).real
        C_trace = np.trace(C_real + C_fake - 2 * C_r_f_sqrt)

        score = U_x_y + C_trace
        return score