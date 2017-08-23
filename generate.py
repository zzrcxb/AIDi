import numpy as np
import keras
import logging
import warnings
import time
import os
import mido

from keras.layers import TimeDistributed, Input, Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, Model

from pre_process import SCNOC_PreProcessor


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

model = keras.models.load_model('test2.h5')
model.load_weights('test2_weights.h5')

timesteps = 40

res = SCNOC_PreProcessor.single_process('test/Forrest Gump.mid', 0)
msgs = SCNOC_PreProcessor.get_msgs('test/Forrest Gump.mid', lambda x: x.type=='note_on')
print(np.array([_[0] for _ in res[0][:timesteps]]).shape)
n_in = np.array([_[0] for _ in res[0][:timesteps]]).reshape(1, timesteps, 128)
# t_in = np.array([_[1] for _ in res[0][:timesteps]]).reshape(1, timesteps, 500)
print(n_in.shape)
# print(t_in.shape)

notes = []
while len(notes) < 50:
    print([i for i, x in enumerate(res[0][timesteps + len(notes)][0]) if x], end='\t')

    predicted = model.predict({'n_in': n_in, })
    pos = predicted[0].flatten().argsort()[-5:][::-1]
    # print(len(predicted))
    nn = predicted[0] > 0.3
    # print(predicted)
    nn = [i for i, x in enumerate(nn) if x]
    notes.append(nn)
    print(nn)

    new_notes = np.zeros(128, dtype=bool)
    for i in nn:
        new_notes[i] = True
    new_notes = new_notes.reshape(1, 1, 128)

    n_in = np.delete(n_in, 0, 1)
    # print(n_in.shape, new_notes.shape)
    n_in = np.append(n_in, new_notes, 1)

# print(notes)
msg_out = msgs[0:40]
for nn in notes:
    if len(nn) == 0:
        continue
    msg_out.append(mido.Message('note_on', note=nn[0], velocity=64, time=249, channel=0))
    del nn[0]
    for _ in nn:
        msg_out.append(mido.Message('note_on', note=_, velocity=64, time=0, channel=0))

mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.extend(msg_out)
mid.save('test.mid')
