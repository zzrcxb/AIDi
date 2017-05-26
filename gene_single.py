from drum import *
from midi_tool import *
from mido import MidiFile, MidiTrack
from numpy.random import choice

model = keras.models.load_model('drum_note4.h5')
model.load_weights('drum_note_weights4.h5')

timesteps = 20
training_set, validation_set, test_set = get_training_data(preprocess('Always_With_Me_Piano_Cover.mid'), 0.3, 0.0, timesteps)
msgs = get_msgs(MidiFile('Always_With_Me_Piano_Cover.mid'), 0, True, note_on=True)

index = 20
seed1 = np.array(validation_set['note']['x'][index]).reshape(1, timesteps, 128)

res = []
for i in range(1000):
    predicted = model.predict(seed1)
    tmp1 = list(predicted.flatten().argsort()[-3:][::-1])

    # tmp1 = choice(tmp1, p=[0.9, 0.05, 0.05])
    tmp1 = tmp1[0]
    res.append(tmp1)

    vec1 = to_categorical(tmp1, 128).reshape(1, 1, 128)

    seed1 = np.delete(seed1, 0, 1)
    seed1 = np.append(seed1, vec1, 1)

print(res)

msg_out = msgs[0:6]
for _ in res:
    msg_out.append(mido.Message('note_on', note=_, velocity=64, time=300, channel=0))

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
track.extend(msg_out)
mid.save('test.mid')