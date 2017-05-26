from drum import *
from midi_tool import *
from mido import MidiFile, MidiTrack
from numpy.random import choice

model = keras.models.load_model('drum_note2.h5')
model.load_weights('drum_note_weights2.h5')

timesteps = 10
training_set, validation_set, test_set = get_training_data(preprocess('Faded.mid'), 0.3, 0.0, timesteps)
msgs = get_msgs(MidiFile('Faded.mid'), 1, True, note_on=True)

index = 100
seed1 = np.array(validation_set['note']['x'][index]).reshape(1, timesteps, 128)
seed2 = np.array(validation_set['velocity']['x'][index]).reshape(1, timesteps, 128)
seed3 = np.array(validation_set['time']['x'][index]).reshape(1, timesteps, 1024)
res = []
for i in range(1000):
    predicted = model.predict(
        {'notes_input': seed1,
        'velocity_input': seed2,
        'times_input': seed3})
    tmp1 = list(predicted[0].flatten().argsort()[-3:][::-1])
    tmp2 = list(predicted[1].flatten().argsort()[-3:][::-1])
    tmp3 = list(predicted[2].flatten().argsort()[-3:][::-1])

    # tmp1 = choice(tmp1, p=[0.9, 0.05, 0.05])
    tmp1 = tmp1[0]
    res.append([tmp1, tmp2[0], tmp3[0]])
    # print(tmp1[0].shape, tmp2[0].shape, tmp3[0].shape)

    vec1 = to_categorical(tmp1, 128).reshape(1, 1, 128)
    vec2 = to_categorical(tmp2[0], 128).reshape(1, 1, 128)
    vec3 = to_categorical(tmp3[0], 1024).reshape(1, 1, 1024)

    seed1 = np.delete(seed1, 0, 1)
    seed2 = np.delete(seed2, 0, 1)
    seed3 = np.delete(seed3, 0, 1)

    seed1 = np.append(seed1, vec1, 1)
    seed2 = np.append(seed2, vec2, 1)
    seed3 = np.append(seed3, vec3, 1)

print(res)

msg_out = msgs[0:6]
for _ in res:
    msg_out.append(mido.Message('note_on', note=_[0], velocity=_[1], time=_[2], channel=1))

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
track.extend(msg_out)
mid.save('test.mid')