import numpy as np
import keras
import logging
import warnings
import time
import os

from keras.layers import TimeDistributed, Input, Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, Model

from pre_process import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings


class Trainer:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.model = None
        self.data_iter = None

    def load_training_data(self, root, channel, count=True):
        pps = SCNOC_PreProcessor(root)
        self.data_iter = pps.data_iter(self.timesteps, channel)
        self.root_path = root
        self.channel = channel
        if count:
            self.data_size = sum(1 for _ in self.data_iter)
            self.data_iter = pps.data_iter(self.timesteps, channel)
            logging.info(str(self.data_size) + ' data loaded.')

    def load_model(self, path):
        self.model = keras.models.load_model(path)
    
    def load_weight(self, path):
        if self.model is not None:
            self.model.load_weights(path)
        else:
            logging.warning('Empty model!')

    def build_SCNOC_model(self, lstm_param, drop_param, dense_param, loss_weights, optimizer='rmsprop'):
        """
        param:
            lstm_param: 2 * 2 list contains integers.
            drop_param: 2 * 2 list contains floats between 0 and 1.
            dense_param: 4 * 2 list contains integers.
            loss_weights: 2 elements list contains floats between 0 and 1.
        """
        notes_in = Input(shape=(self.timesteps, PITCH_LEVELS), name='n_in')
        # times_in = Input(shape=(self.timesteps, TICK_MAX), name='t_in')

        level = 0

        n_lstm_o1 = LSTM(lstm_param[level][0], return_sequences=True, dropout=drop_param[level][0])(notes_in)
        # t_lstm_o1 = LSTM(lstm_param[level][1], return_sequences=True, dropout=drop_param[level][1])(times_in)

        level = 1

        n_lstm_o2 = LSTM(lstm_param[level][0], return_sequences=False, dropout=drop_param[level][0])(n_lstm_o1)
        # t_lstm_o2 = LSTM(lstm_param[level][0], return_sequences=False, dropout=drop_param[level][0])(t_lstm_o1)

        # merged = keras.layers.concatenate([n_lstm_o2, t_lstm_o2])
        merged = n_lstm_o2

        level = 0
        notes = Dense(dense_param[level][0], activation='sigmoid')(merged)
        # times = Dense(dense_param[level][1], activation='sigmoid')(merged)

        level = 1
        notes = Dense(dense_param[level][0], activation='sigmoid')(notes)
        # times = Dense(dense_param[level][1], activation='sigmoid')(times)

        level = 2
        notes = Dense(dense_param[level][0], activation='sigmoid')(notes)
        # times = Dense(dense_param[level][1], activation='sigmoid')(times)

        level = 3
        n_out = Dense(dense_param[level][0], activation='sigmoid', name='n_out')(notes)
        # t_out = Dense(dense_param[level][1], activation='sigmoid', name='t_out')(times)

        # model = Model(inputs=[notes_in, times_in], outputs=[n_out, t_out])
        model = Model(inputs=[notes_in, ], outputs=[n_out, ])

        logging.info('Compiling model with ' + optimizer + ' optimizer. ' + 'loss_weights = ' + repr(loss_weights))

        # model.compile(optimizer=optimizer, loss={'n_out': 'binary_crossentropy', 't_out': 'categorical_crossentropy'},
            # loss_weights=loss_weights, metrics=['accuracy'])
        model.compile(optimizer=optimizer, loss={'n_out': 'binary_crossentropy', }, metrics=['accuracy'])

        logging.info('Compiled model...')

        self.model = model
        return model

    def iter_wrapper(self, batches):
        notes_x = []
        notes_y = []
        times_x = []
        times_y = []
        # batches = self.data_size // batch_size
        # remains = self.data_size % batch_size
        batch_size = self.data_size // batches
        # if remains:
        #     batches += 1
        while True:
            self.load_training_data(self.root_path, self.channel, False)
            for i in range(batches):
                if i != batches - 1:
                    for j in range(batch_size):
                        data = next(self.data_iter)
                        notes_x.append(data[0][0])
                        notes_y.append(data[1][0])
                        times_x.append(data[0][1])
                        times_y.append(data[1][1])
                else:
                    for data in self.data_iter:
                        notes_x.append(data[0][0])
                        notes_y.append(data[1][0])
                        times_x.append(data[0][1])
                        times_y.append(data[1][1])
                nx = np.array(notes_x)
                tx = np.array(times_x)
                ny = np.array(notes_y)
                ty = np.array(times_y)
                notes_x = []
                notes_y = []
                times_x = []
                times_y = []
                yield [nx, ], [ny, ]

    def train(self, epochs, batches, prefix='training'):
        if self.model is None or self.data_iter is None:
            logging.error('Empty model or empty dataset!')
            return None

        self.model.fit_generator(
            self.iter_wrapper(batches),
            batches,
            epochs=epochs,
            # use_multiprocessing=True
            # batch_size=batch_size
        )

        self.model.save(prefix + '.h5')
        self.model.save_weights(prefix + '_weights.h5')


if __name__ == '__main__':
    from universal import *
    init_all()
    test = Trainer(40)
    test.load_training_data('test', 0)
    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.5, decay=0.1, nesterov=False)
    test.build_SCNOC_model([[64, 64], [64, 64]], [[0.2, 0.2], [0.2, 0.2]],
     [[64, 64], [64, 128], [128, 256], [128, 500]], [1, 1], optimizer='adam')
    # test.load_model('test1.h5')
    # test.load_weight('test1_weights.h5')
    test.train(200, 30, 'test3')
    # for i, data in enumerate(test.iter_wrapper(30)):
    #     print(i)
