import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
import os
import warnings

from keras.layers import TimeDistributed, Input, Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop


from midi_tool import *
from mido import Message


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings


def preprocess(path):
    msgs = get_msgs(mido.MidiFile(path), 0, True, note_on=True)
    snd = list(map(Message.dict, msgs))
    notes = [_['note'] for _ in snd]
    times = [_['time'] for _ in snd]
    velocity = [_['velocity'] for _ in snd]
    return notes, times, velocity

def preprocess_multi(path, filenames=None):
    notes, times, velocities = [], [], []
    for file in os.listdir(path):
        if filenames != None and file not in filenames:
            continue
        notes_tmp, times_tmp, vel_tmp = preprocess(os.path.join(path, file))
        notes.extend(notes_tmp)
        times.extend(times_tmp)
        vel_tmp.extend(velocities)
    return notes, times, velocities


def get_training_data(msgs, validation_ratio, test_ratio, batch_size, max_time=1024):
    notes, times, velocity = msgs
    # Clear first time
    times[0] = 0

    bnotes, btimes, bvels = [], [], []
    lnotes, ltimes, lvels = [], [], []
    for i in range(len(notes) - batch_size):
        if max(times[i: i + batch_size]) >= max_time or times[i + batch_size] >= max_time:
            # print(max(times[i: i + batch_size]), times[i + batch_size])
            continue

        bnotes.append(np.array(notes[i: i + batch_size]))
        btimes.append(times[i: i + batch_size])
        bvels.append(velocity[i: i + batch_size])

        lnotes.append(to_categorical(notes[i + batch_size], 128).flatten().astype(bool))
        ltimes.append(to_categorical(times[i + batch_size], max_time).flatten().astype(bool))
        lvels.append(to_categorical(velocity[i + batch_size], 128).flatten().astype(bool))

    vnotes, vtimes, vvels = [], [], []
    for seqs in bnotes:
        tmp = []
        for i in seqs:
            tmp.append(to_categorical(i, 128).flatten().astype(bool))
        vnotes.append(tmp)

    for seqs in bvels:
        tmp = []
        for i in seqs:
            tmp.append(to_categorical(i, 128).flatten().astype(bool))
        vvels.append(tmp)
    
    for seqs in btimes:
        tmp = []
        for i in seqs:
            tmp.append(to_categorical(i, max_time).flatten().astype(bool))
        vtimes.append(tmp)

    # Get each set
    validation_size = int(validation_ratio * len(vnotes))
    test_size = int(test_ratio * len(vnotes))
    train_size = len(vnotes) - validation_size - test_size

    indexs = list(range(len(vnotes)))
    random.shuffle(indexs)
    train_index = indexs[0: train_size]
    validation_index = indexs[train_size: train_size + validation_size]
    test_index = indexs[train_size + validation_size: len(vnotes)]

    training_set = {'note': {'x': [vnotes[_] for _ in train_index], 'y': [lnotes[_] for _ in train_index]}, 
                    'time': {'x': [vtimes[_] for _ in train_index], 'y': [ltimes[_] for _ in train_index]}, 
                    'velocity': {'x': [vvels[_] for _ in train_index], 'y': [lvels[_] for _ in train_index]}}
    
    validation_set = {'note': {'x': [vnotes[_] for _ in validation_index], 'y': [lnotes[_] for _ in validation_index]}, 
                      'time': {'x': [vtimes[_] for _ in validation_index], 'y': [ltimes[_] for _ in validation_index]}, 
                      'velocity': {'x': [vvels[_] for _ in validation_index], 'y': [lvels[_] for _ in validation_index]}}

    test_set = {'note': {'x': [vnotes[_] for _ in test_index], 'y': [lnotes[_] for _ in test_index]}, 
                'time': {'x': [vtimes[_] for _ in test_index], 'y': [ltimes[_] for _ in test_index]}, 
                'velocity': {'x': [vvels[_] for _ in test_index], 'y': [lvels[_] for _ in test_index]}}

    return training_set, validation_set, test_set


def build_model(timesteps, dim):
    model = Sequential()
    model.add(LSTM(dim, input_shape=(timesteps, dim), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(dim * 2, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(dim * 2))
    model.add(Activation('sigmoid'))

    model.add(Dense(dim * 2))
    model.add(Activation('sigmoid'))

    model.add(Dense(dim))
    model.add(Activation('softmax'))

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)
    return model

def build_full_model(timesteps, dims):
    # Inputs
    notes_input = Input(shape=(timesteps, dims[0]), name='notes_input')
    velocity_input = Input(shape=(timesteps, dims[1]), name='velocity_input')
    times_input = Input(shape=(timesteps, dims[2]), name='times_input')

    # Note layer
    # LSTM
    notes_lstm_out1 = LSTM(dims[0], return_sequences=True, dropout=0.2)(notes_input)
    notes_lstm_out = LSTM(dims[0], return_sequences=False, dropout=0.2)(notes_lstm_out1)

    # Dense
    notes = Dense(dims[0] * 2, activation='relu')(notes_lstm_out)
    notes = Dense(dims[0] * 2, activation='sigmoid')(notes)
    notes = Dense(dims[0], activation='sigmoid')(notes)
    notes_out = Dense(dims[0], activation='softmax', name='notes_output')(notes)

    # Velocity
    velocities = LSTM(dims[1], return_sequences=True, dropout=0.2)(velocity_input)
    velocities = keras.layers.concatenate([velocities, notes_lstm_out1])
    velocities = LSTM(dims[1], return_sequences=False, dropout=0.2)(velocities)
    velocities = Dense(dims[1] * 2, activation='relu')(velocities)
    velocities = Dense(dims[1] * 2, activation='sigmoid')(velocities)
    velocities = Dense(dims[1], activation='sigmoid')(velocities)
    velocities_out = Dense(dims[1], activation='softmax', name='velocity_out')(velocities)

    # Times
    times = LSTM(dims[2], return_sequences=True, dropout=0.2)(times_input)
    times = keras.layers.concatenate([times, notes_lstm_out1])
    times = LSTM(dims[2], return_sequences=False, dropout=0.2)(times)
    times = Dense(dims[2] * 2, activation='relu')(times)
    times = Dense(dims[2] * 2, activation='sigmoid')(times)
    times = Dense(dims[2], activation='sigmoid')(times)
    times_out = Dense(dims[2], activation='softmax', name='times_out')(times)

    model = Model(inputs=[notes_input, velocity_input, times_input], outputs=[notes_out, velocities_out, times_out])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', loss_weights=[1., 0.3, 0.5], metrics=['accuracy'])
    return model


def training(path):
    timesteps = 20
    training_set, validation_set, test_set = get_training_data(preprocess_multi(path), 0.3, 0.0, timesteps)

    x1data = np.array(training_set['note']['x'])
    y1data = np.array(training_set['note']['y'])
    x2data = np.array(training_set['velocity']['x'])
    y2data = np.array(training_set['velocity']['y'])
    x3data = np.array(training_set['time']['x'])
    y3data = np.array(training_set['time']['y'])
    model = build_full_model(timesteps, [128, 128, 1024])
    # model = build_model(timesteps, 128)

    print(x1data.shape)
    print(y1data.shape)

    model.fit(
        {'notes_input': x1data, 'velocity_input': x2data, 'times_input': x3data},
        {'notes_output': y1data, 'velocity_out': y2data, 'times_out': y3data},
        epochs=20,
        batch_size=128)

    # model.fit(
    #     x1data,
    #     y1data,
    #     epochs=300,
    #     batch_size=128)

    model.save('drum_note4.h5')
    model.save_weights('drum_note_weights4.h5')

    return model


def main():
    model = training("./drum")


if __name__ == '__main__':
    main()
