# code heavily inspired from NVIDIA/mellotron/mellotron_utils.py
import numpy as np
import re
import torch

_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

########################
#  CONSONANT DURATION  #
########################
PHONEMEDURATION = {
    'B': 0.05,
    'CH': 0.1,
    'D': 0.075,
    'DH': 0.05,
    'DX': 0.05,
    'EL': 0.05,
    'EM': 0.05,
    'EN': 0.05,
    'F': 0.1,
    'G': 0.05,
    'HH': 0.05,
    'JH': 0.05,
    'K': 0.05,
    'L': 0.05,
    'M': 0.15,
    'N': 0.15,
    'NG': 0.15,
    'NX': 0.05,
    'P': 0.05,
    'Q': 0.075,
    'R': 0.05,
    'S': 0.1,
    'SH': 0.05,
    'T': 0.075,
    'TH': 0.1,
    'V': 0.05,
    'Y': 0.05,
    'W': 0.05,
    'WH': 0.05,
    'Z': 0.05,
    'ZH': 0.05
}

valid_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

_arpabet = ['@' + s for s in valid_symbols]

_punctuation = '!\'",.:;? '
_math = '#%&*+-/[]()'
_special = '_@©°½—₩€$'
_accented = 'áçéêëñöøćž'
_numbers = '0123456789'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

symbols = list(_punctuation + _math + _special + _accented + _numbers + _letters) + _arpabet

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence(text):
    sequence = []

    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(text)
            break

        sequence += text_to_sequence(m.group(1))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence

def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'


def add_space_between_events(events, connect=False):
    new_events = []
    for i in range(1, len(events)):
        token_a, freq_a, start_time_a, end_time_a = events[i-1][-1]
        token_b, freq_b, start_time_b, end_time_b = events[i][0]

        if token_a in (' ', '') and len(events[i-1]) == 1:
            new_events.append(events[i-1])
        elif token_a not in (' ', '') and token_b not in (' ', ''):
            new_events.append(events[i-1])
            if connect:
                new_events.append([[' ', 0, end_time_a, start_time_b]])
            else:
                new_events.append([[' ', 0, end_time_a, end_time_a]])
        else:
            new_events.append(events[i-1])

    if new_events[-1][0][0] != ' ':
        new_events.append([[' ', 0, end_time_a, end_time_a]])
    new_events.append(events[-1])

    return new_events

def adjust_extensions(events, phoneme_durations):
    if len(events) == 1:
        return events

    idx_last_vowel = None
    n_consonants_after_last_vowel = 0
    rest_after_last_vowel = False
    target_ids = np.arange(len(events))
    for i in range(len(events)):
        token = re.sub('[0-9{}]', '', events[i][0])
        if idx_last_vowel is None and token not in phoneme_durations:
            idx_last_vowel = i
            n_consonants_after_last_vowel = 0
        else:
            if token == '_' and not n_consonants_after_last_vowel:
                events[i][0] = events[idx_last_vowel][0]
            elif token == '_' and n_consonants_after_last_vowel:
                events[i][0] = events[idx_last_vowel][0]
                start = idx_last_vowel + 1
                target_ids[start:start+n_consonants_after_last_vowel] += 1 + int(rest_after_last_vowel)
                target_ids[i] -= n_consonants_after_last_vowel
                if rest_after_last_vowel:
                    target_ids[i-1] -= n_consonants_after_last_vowel
            elif token in phoneme_durations:
                n_consonants_after_last_vowel += 1
            elif token == ' ':
                rest_after_last_vowel = True
            else:
                rest_after_last_vowel = False
                n_consonants_after_last_vowel = 0
                idx_last_vowel = i

    new_events = [0] * len(events)
    for i in range(len(events)):
        new_events[target_ids[i]] = events[i]

    # adjust time of consonants that were repositioned
    for i in range(1, len(new_events)):
        if new_events[i][2] < new_events[i-1][2]:
            new_events[i][2] = new_events[i-1][2]
            new_events[i][3] = new_events[i-1][3]
        if new_events[i][0][0] == '{':
            new_events[i][0] = new_events[i][0][1:]
        if new_events[i][0][-1] == '}' and i < len(new_events) - 1:
            new_events[i][0] = new_events[i][0][:-1]

    first_p = new_events[0][0]
    if not first_p.isspace() and first_p[0] != '{':
        new_events[0][0] = '{' + first_p
    last_p = new_events[-1][0]
    if not last_p.isspace() and last_p[-1] != '}':
        new_events[-1][0] = last_p + '}'
    return new_events


def adjust_consonant_lengths(events, phoneme_durations):
    t_init = events[0][2]
    t_end = events[-1][3]
    duration = t_end - t_init
    consonant_durations = {}
    consonant_duration = 0
    for event in events:
        c = re.sub('[0-9{}]', '', event[0])
        if c in phoneme_durations:
            consonant_durations[c] = phoneme_durations[c]
            consonant_duration += phoneme_durations[c]

    if not consonant_duration <= 0.4 * duration:
        scale = 0.4 * duration / consonant_duration
        for k, v in consonant_durations.items():
            consonant_durations[k] = scale * v

    idx_last_vowel = None
    for i in range(len(events)):
        task = re.sub('[0-9{}]', '', events[i][0])
        if task in consonant_durations:
            duration = consonant_durations[task]
            if idx_last_vowel is None:  # consonant comes before any vowel
                events[i][2] = t_init
                events[i][3] = t_init + duration
            else:  # consonant comes after a vowel, must offset
                events[idx_last_vowel][3] -= duration
                for k in range(idx_last_vowel+1, i):
                    events[k][2] -= duration
                    events[k][3] -= duration
                events[i][2] = events[i-1][3]
                events[i][3] = events[i-1][3] + duration
        else:
            events[i][2] = t_init
            events[i][3] = events[i][3]
            t_init = events[i][3]
            idx_last_vowel = i
        t_init = events[i][3]

    return events


def adjust_consonants(events, phoneme_durations):
    if len(events) == 1:
        return events

    start = 0
    split_ids = []
    t_init = events[0][2]

    # get each substring group
    for i in range(1, len(events)):
        if events[i][2] != t_init:
            split_ids.append((start, i))
            start = i
            t_init = events[i][2]
    split_ids.append((start, len(events)))

    for (start, end) in split_ids:
        events[start:end] = adjust_consonant_lengths(
            events[start:end], phoneme_durations)

    return events


def event2alignment(events, hop_length=256, sampling_rate=22050):
    frame_length = float(hop_length) / float(sampling_rate)

    n_frames = int(events[-1][-1][-1] / frame_length)
    n_tokens = np.sum([len(e) for e in events])
    alignment = np.zeros((n_tokens, n_frames))

    cur_event = -1
    for event in events:
        for i in range(len(event)):
            if len(event) == 1 or cur_event == -1 or event[i][0] != event[i-1][0]:
                cur_event += 1
            token, freq, start_time, end_time = event[i]
            alignment[cur_event, int(start_time/frame_length):int(end_time/frame_length)] = 1

    return alignment[:cur_event+1]


def event2f0(events, hop_length=256, sampling_rate=22050):
    frame_length = float(hop_length) / float(sampling_rate)
    n_frames = int(events[-1][-1][-1] / frame_length)
    f0s = np.zeros((1, n_frames))

    for event in events:
        for i in range(len(event)):
            token, freq, start_time, end_time = event[i]
            f0s[0, int(start_time/frame_length):int(end_time/frame_length)] = freq

    return f0s


def event2text(events, convert_stress):
    text_clean = ''
    for event in events:
        for i in range(len(event)):
            if i > 0 and event[i][0] == event[i-1][0]:
                continue
            if event[i][0] == ' ' and len(event) > 1:
                if text_clean[-1] != "}":
                    text_clean = text_clean[:-1] + '} {'
                else:
                    text_clean += ' {'
            else:
                if event[i][0][-1] in ('}', ' '):
                    text_clean += event[i][0]
                else:
                    text_clean += event[i][0] + ' '

    if convert_stress:
        text_clean = re.sub('[0-9]', '1', text_clean)

    text_encoded = text_to_sequence(text_clean)
    return text_encoded, text_clean


def remove_excess_frames(alignment, f0s):
    excess_frames = np.sum(alignment.sum(0) == 0)
    alignment = alignment[:, :-excess_frames] if excess_frames > 0 else alignment
    f0s = f0s[:, :-excess_frames] if excess_frames > 0 else f0s
    return alignment, f0s


def get_data_from_text_events_with_phonemes(text_events, ticks=False, tempo=120, resolution=960, phoneme_durations=None, convert_stress=False):
    def pitch_to_freq(pitch):
        return 440*(2**((pitch - 69)/12))

    if ticks:
        num = int
        to_time = lambda t: t * 60/(tempo*resolution)
    else:
        num = float
        to_time = lambda t: t

    if phoneme_durations is None:
        phoneme_durations = PHONEMEDURATION
    
    events = []
    phonemes = ''
    word = []
    time = 0
    note_off = True
    rest_start = -1
    new_word = True
    for e in text_events:
        e_split = e.split('_')
        if '_' not in e:
            phonemes = e
        elif e == '_R_':
            phonemes = '_'
        elif e_split[0] == 'ON':
            if rest_start >= 0:
                if new_word and word:
                    last_p = word[-1][0]
                    if last_p != ' ':
                        word[-1][0] = last_p + '}'
                    events.append(word)
                    word = []
                word.append([' ', 0, to_time(rest_start), to_time(time)])
                rest_start = -1
            freq = pitch_to_freq(int(e_split[1]))
            start = time
            note_off = False
        elif e_split[0] == 'W':
            t = num(e_split[1])
            if note_off and rest_start < 0:
                rest_start = time
            time += t
        elif e == '_OFF_':
            if new_word and word:
                last_p = word[-1][0]
                if last_p != ' ':
                    word[-1][0] = last_p + '}'
                events.append(word)
                word = []            
            for i, p in enumerate(phonemes.split()):
                if new_word and i == 0:
                    word.append(['{' + p, freq, to_time(start), to_time(time)])
                else:
                    word.append([p, freq, to_time(start), to_time(time)])
            new_word = False
            note_off = True
        else:
            new_word = True

    last_p = word[-1][0]
    if last_p != ' ':
        word[-1][0] = last_p + '}'
    events.append(word)

    # make adjustments
    events = [adjust_extensions(e, phoneme_durations)
                      for e in events]
    events = [adjust_consonants(e, phoneme_durations)
                      for e in events]
    events = add_space_between_events(events)

    # convert data to alignment, f0 and text encoded
    alignment = event2alignment(events)
    f0s = event2f0(events)
    alignment, f0s = remove_excess_frames(alignment, f0s)
    text_encoded, text_clean = event2text(events, convert_stress)

    # convert data to torch
    alignment = torch.from_numpy(alignment).permute(1, 0)[:, None].float()
    f0s = torch.from_numpy(f0s)[None].float()
    text_encoded = torch.LongTensor(text_encoded)[None]

    return {'rhythm': alignment, 'pitch_contour': f0s, 'text_encoded': text_encoded}
