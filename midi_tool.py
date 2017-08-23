import mido
import time
from mido import MidiFile, MidiTrack


def play_midi(path, channel=None, show=False):
    with mido.open_output() as outport:
        start = time.time()
        for msg in mido.MidiFile(path).play():
            if channel == None:
                outport.send(msg)
                if show:
                    print(msg)
            elif not msg.is_meta and msg.type == 'note_on' and msg.channel == channel:
                outport.send(msg)
                if show:
                    print(msg)
            print('\r%.1fs' % (time.time() - start), end='')


def get_msgs(midi_f, channel=None, convert=None, tempo=500000, show=False, note_on=False):
    dTime = 0
    msgs = []
    ticks_per_beat = midi_f.ticks_per_beat
    for msg in midi_f:
        if channel == None:
            if convert:
                msg.time = int(mido.second2tick(msg.time, ticks_per_beat, tempo))
            msgs.append(msg)
            if show:
                print(msg)
        else:
            if note_on and msg.type != 'note_on':
                continue
            if not msg.is_meta and msg.channel == channel:
                msg.time += dTime
                if convert:
                    msg.time = int(mido.second2tick(msg.time, ticks_per_beat, tempo))
                msgs.append(msg)
                dTime = 0
                if show:
                    print(msg)
            else:
                dTime += msg.time
    return msgs


def get_channels(dimi_f):
    channels = []
    for msg in dimi_f:
        if not msg.is_meta and msg.channel not in channels:
            channels.append(msg.channel)
    return channels


def save2file(msgs, path):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.extend(msgs)
    mid.save(path)
