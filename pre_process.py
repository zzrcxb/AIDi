import mido
import os
import numpy as np
import logging


PITCH_LEVELS = 128
VELOCITY_LEVELS = 128
TICK_MAX = 800


class PreProcessor:
    def __init__(self, root):
        self.root_path = root

    def process_all(self, channel=None):
        pass

    @staticmethod
    def time_converter(msg_time, ticks_per_beat, tempo=500000):
        return int(mido.second2tick(msg_time, ticks_per_beat, tempo))

    @staticmethod
    def get_msgs(path, ft=None, convert_tempo=True, tempo=500000, get_channels=False):
        midiF = mido.MidiFile(path)
        res = list(filter(ft, midiF))

        if convert_tempo:
            for _ in res:
                _.time = PreProcessor.time_converter(_.time, midiF.ticks_per_beat, tempo)

        if get_channels:
            channels = {msg.channel: msg.program for msg in filter(lambda x: x.type=='program_change', midiF)}
            return res, channels
        else:
            return res


# Single Channel Note_on Only Chord support(SCNOC) pre-processor
class SCNOC_PreProcessor(PreProcessor):
    def __init__(self, root):
        PreProcessor.__init__(self, root)

    @classmethod
    def single_process(cls, path, channel):
        def filt(msg):
            if msg.type == 'note_on' and not msg.is_meta and msg.channel == channel:
                return True
            else:
                return False

        res = PreProcessor.get_msgs(path, filt)

        logging.debug('Read msgs: ' + repr(res))

        if len(res) == 0:
            return []

        # Split sequence into seperated pieces
        pitches = []
        curChord = [[res[0], ], res[0].time]
        del res[0]
        for msg in res:
            if msg.time == 0:
                curChord[0].append(msg)
            else:
                pitches.append(curChord)
                curChord = [[msg, ], msg.time]

        logging.debug(repr(pitches))

        # Convert pieces into array format
        pro_res = []
        for click in pitches:
            notes = np.zeros(PITCH_LEVELS, dtype=bool)
            for msg in click[0]:
                notes[msg.note] = True
            pro_res.append([notes, click[1]])

        # Remove too much long ticks
        div = [0, ]
        for index, msgs in enumerate(pro_res):
            if msgs[1] >= TICK_MAX and index != 0:
                div.append(index)
        div += [len(pro_res), ]

        results = []
        for i, j in zip(div[:-1], div[1:]):

            tmp = pro_res[i : j]
            tmp[0][1] = 0
            for _ in tmp:
                times = np.zeros(TICK_MAX, dtype=bool)
                times[_[1]] = True
                _[1] = times

            results.append(tmp)

        return results

    def data_iter(self, timesteps, channel):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if not file.endswith('.mid'):
                    continue

                # print(os.path.join(root, file))
                results = SCNOC_PreProcessor.single_process(os.path.join(root, file), channel)

                for result in results:
                    if len(result) < timesteps + 1:
                        continue
                    for index in range(len(result) - timesteps - 1):
                        notes = [_[0] for _ in result[index: index + timesteps + 1]]
                        times = [_[1] for _ in result[index: index + timesteps + 1]]

                        yield [np.array(notes[:-1]), np.array(times[:-1])], [notes[-1], times[-1]]


def tests():
    pps = SCNOC_PreProcessor('./piano')

    cnt = 0
    it = pps.data_iter(50, 0)

    print(next(it))
    res = []
    note = [0, 45, 89]
    for i in range(5):
        res.append(mido.Message('note_on', note=127, time=30))
        for j in range(3):
            msg = mido.Message('note_on', note=note[j], time=0)
    
    res.append(mido.Message('note_on', note=127, time=800))
    note2 = [4, 48, 9]
    for i in range(5):
        res.append(mido.Message('note_on', note=127, time=60))
        for j in range(3):
            msg = mido.Message('note_on', note=note2[j], time=0)

    res = SCNOC_PreProcessor.single_process('', 1)
    for piece in res:
        for i in piece:
            print(np.argmax(i[1]))
        print("End of piece")

if __name__ == '__main__':
    tests()
