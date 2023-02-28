import argparse
import sys
import numpy as np
import random
import time
import os
from keras.layers import *
import tensorflow as tf
from subprocess import check_output
import h5py
import re
import math
import pandas as pd
from os.path import splitext, basename, isfile
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, BatchNormalization, Dropout, Activation, merge, Conv2D, \
    MaxPooling2D, Activation, LeakyReLU, concatenate
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from itertools import combinations
import bisect
from attention import cbam

random.seed(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
class GeLU(Activation):
    def __init__(self, activation, **kwargs):
        super(GeLU, self).__init__(activation, **kwargs)
        self.__name__ = 'gelu'


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'gelu': GeLU(gelu)})

class CBAM_GAN():
    def __init__(self, datasets, n_latent_dim,weight=0.001, model_path='CBAM_GAN.h5', epochs=100, batch_size=64):
        self.latent_dim = n_latent_dim
        optimizer = Adam()
        self.n = len(datasets)
        self.epochs = epochs
        self.batch_size = batch_size
        sample_size = 0
        if self.n > 1:
            sample_size = datasets[0].shape[0]
        print(sample_size)
        if sample_size > 300:
            self.epochs = 11
        else:
            self.epochs = 10
        self.epochs = 30 * batch_size
        self.shape = []
        self.weight = [0.3,0.5,0.2]
        self.disc_w = 1e-4
        self.model_path = model_path
        input = []
        loss = []
        loss_weights = []
        output = []
        for i in range(self.n):
            self.shape.append(datasets[i].shape[1])
            loss.append('mse')
        loss.append('binary_crossentropy')
        self.decoder, self.disc = self.build_decoder_disc()
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.encoder = self.build_encoder()
        self.disc.trainable = False
        for i in range(self.n):
            input.append(Input(shape=(self.shape[i],)))
            loss_weights.append((1 - self.disc_w) * self.weight[i])
        loss_weights.append(self.disc_w)
        z_mean, z_log_var, z = self.encoder(input)
        output = self.decoder(z)
        self.gan = Model(input, output)
        self.gan.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
        print(self.gan.summary())
        return

    def build_encoder(self):
        def sampling(args):
            z_mean, z_log_var = args
            return z_mean + K.exp(0.5 * z_log_var) * K.random_normal(K.shape(z_mean), seed=0)

        encoding_dim = self.latent_dim
        X = []
        dims = []
        denses = []
        for i in range(self.n):
            X.append(Input(shape=(self.shape[i],)))
            dims.append(int(encoding_dim * self.weight[i]))
        for i in range(self.n):
            denses.append(Dense(dims[i])(X[i]))
        if self.n > 1:
            merged_dense = concatenate(denses, axis=-1)
        else:
            merged_dense = denses[0]
        model = BatchNormalization()(merged_dense)
        model = Activation('gelu')(model)
        model = Dense(256, activation="gelu")(model)
        model = cbam(model)
        model = Dense(encoding_dim)(model)
        z_mean = Dense(encoding_dim)(model)
        z_log_var = Dense(encoding_dim)(model)
        z = Lambda(sampling, output_shape=(encoding_dim,), name='z')([z_mean, z_log_var])
        return Model(X, [z_mean, z_log_var, z])

    def build_decoder_disc(self):
        denses = []
        X = Input(shape=(self.latent_dim,))
        model =Dense(self.latent_dim)(X)
        model = Dense(256, activation="gelu")(model)
        model = cbam(model)
        model = BatchNormalization()(model)
        model = Activation('gelu')(model)
        for i in range(self.n):
            denses.append(Dense(self.shape[i])(model))
        dec = Dense(1, activation='sigmoid')(model)
        denses.append(dec)
        m_decoder = Model(X, denses)
        m_disc = Model(X, dec)
        return m_decoder, m_disc

    def build_disc(self):
        x = Input(shape=(self.latent_dim,))
        dec = Dense(1, activation='sigmoid', kernel_initializer="glorot_normal")(x)
        output = Model(x, dec)
        return output


    def train(self, X_train, bTrain=True):
        model_path = self.model_path
        log_file = "./run.log"
        fp = open(log_file, 'w')
        if bTrain:
            # GAN
            valid = np.ones((self.batch_size, 1))
            fake = np.zeros((self.batch_size, 1))
            for epoch in range(self.epochs):
                #  Train Discriminator
                data = []
                idx = np.random.randint(0, X_train[0].shape[0], self.batch_size)
                for i in range(self.n):
                    data.append(X_train[i][idx])
                latent_fake = self.encoder.predict(data)[2]

                latent_real = np.random.normal(size=(self.batch_size, self.latent_dim))
                d_loss_real = self.disc.train_on_batch(latent_real,valid)
                d_loss_fake = self.disc.train_on_batch(latent_fake,fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                outs = data + [valid]
                #  Train Encoder_GAN
                g_loss = self.gan.train_on_batch(data, outs)
                print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
            fp.close()
            self.encoder.save(model_path)
        else:
            self.encoder = load_model(model_path)
        mat = self.encoder.predict(X_train)[0]
        return mat

class GAN_API(object):
    def __init__(self, model_path='./model/', epochs=200, weight=0.001):
        self.model_path = model_path
        self.score_path = './score/'
        self.epochs = epochs
        self.batch_size = 16
        self.weight = weight

    # feature extract
    def feature_gan(self, datasets, index=None, n_components=140, b_decomposition=True, weight=0.001):
        if b_decomposition:
            X = self.encoder_gan(datasets, n_components)
            fea = pd.DataFrame(data=X, index=index, columns=map(lambda x: 'v' + str(x), range(X.shape[1])))
        else:
            fea = np.concatenate(datasets)
        print("feature extract finished!")
        return fea
    def encoder_gan(self, ldata, n_components=140):
        egan = CBAM_GAN(ldata, n_components, self.weight, self.model_path, self.epochs, self.batch_size)
        return egan.train(ldata)

def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='CBAM_GAN v1.0')
    parser.add_argument("-i", dest='file_input', default="/Users/limingna/PycharmProjects/p1/CBAM-GAN/inputs/KIRC.list",
                        help="file input")
    parser.add_argument("-e", dest='epochs', type=int, default=200, help="Number of iterations")
    parser.add_argument("-m", dest='run_mode', default="CBAM_GAN", help="run_mode: feature")
    parser.add_argument("-w", dest='disc_weight', type=float, default=1e-4, help="weight")
    parser.add_argument("-o", dest='output_path', default="./score/", help="file output")
    parser.add_argument("-t", dest='type',default="KIRC", help="cancer type: LGG, GBM")
    args = parser.parse_args()
    model_path = './model/' + args.type + '.h5'
    GAN = GAN_API(model_path, epochs=args.epochs, weight=args.disc_weight)

    if args.run_mode == 'CBAM_GAN':
        cancer_type = args.type
        fea_tmp_file = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/' + cancer_type + '.csv'
        tmp_dir = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/' + cancer_type + '/'
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        ldata = []
        l = []
        #
        for line in open(args.file_input, 'rt'):
            base_file = splitext(basename(line.rstrip()))[0]
            fea_save_file = tmp_dir + base_file + '.csv'

            df_new = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
            l = list(df_new)
            df_new = df_new.T
            ldata.append(df_new.values.astype(float))
        start_time = time.time()
        vec = GAN.feature_gan(ldata, index=l, n_components=140, weight=args.disc_weight)
        df = pd.DataFrame(data=[time.time() - start_time])
        vec.to_csv(fea_tmp_file, header=True, index=True, sep='\t')
        out_file = '/Users/limingna/PycharmProjects/p1/CBAM-GAN/result/' + cancer_type + '.time'
        df.to_csv(out_file, header=True, index=False, sep=',')



if __name__ == "__main__":
    main()