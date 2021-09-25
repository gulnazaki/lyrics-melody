import pretty_midi
from utils import *
from heapq import merge
from collections import OrderedDict
from phonetisaurus import Phonetisaurus
import numpy as np
from scipy.io.wavfile import write
import os
import tempfile
import music21

class Midi(pretty_midi.PrettyMIDI):
	def __init__(self, midi_file=None, norm_resolution=None, upper_text=False, phonemodel='cmudict-20170708.o8.fst', music_analysis=False):
		super().__init__(midi_file)
		
		if midi_file is not None:
			self.midi_file = midi_file
			self.norm_time_to_tick = lambda t: round(self.time_to_tick(t) * norm_resolution / self.resolution) if norm_resolution else self.time_to_tick(t)
			self.norm_tick_to_time = lambda t: (self.tick_to_time(t) * self.resolution / norm_resolution) if norm_resolution else self.tick_to_time(t)

			self.lyrics_margin = lyrics_margin(self.get_tempo_changes()[1])
			self._reclassify_lyrics_and_text()
			self._sort_notes_and_lyrics()
			self.lyrics = format_lyrics(self.lyrics, upper_text=upper_text)
			self.__lyrics_times = OrderedDict([(l.time, l) for l in self.lyrics])
			## removing duplicate lyrics
			self.lyrics = list(self.__lyrics_times.values())

			self.is_instrumental = True
			self.vocal_instrument = None
			self.other_instruments = self.instruments
			self.lyrics_sung = []
			if len(self.lyrics) >= LYRICS_THRESH:
				self._find_vocal_instrument()
				if not self.is_instrumental:
					self._create_sung_lyrics()
					self._trim_overlapped_vocals()
					self._syllablify(Phonetisaurus(phonemodel))
			if music_analysis:
				self.key = self.get_music21_score([i for i in self.instruments if not i.is_drum]).analyze('key')
			else:
				self.key = None

	def _reclassify_lyrics_and_text(self):
		fix_special_chars(self.lyrics)
		fix_special_chars(self.text_events)
		valid_lyrics = [l for l in self.lyrics if valid_lyric(l.time, l.text)]
		valid_text_events = [t for t in self.text_events if valid_lyric(t.time, t.text)]

		if len(valid_lyrics) > len(valid_text_events):
			self.lyrics = valid_lyrics
		else:
			self.lyrics = [pretty_midi.Lyric(t.text, t.time) for t in valid_text_events]

	def _sort_notes_and_lyrics(self):
		self.lyrics.sort(key=lambda x: x.time)
		for i in self.instruments:
			i.notes.sort(key=lambda x: x.start)

	def _find_vocal_instrument(self):
		min_offset = None
		max_matches = len(self.lyrics) * VALID_THRESH
		for instrument in self.instruments:
			if instrument.is_drum:
				continue
			note_dict = OrderedDict([(n.start, n) for n in instrument.notes])
			offset, matches = match_lyrics(list(self.__lyrics_times.keys()),
											list(note_dict.keys()),
											self.lyrics_margin)
			
			if abs(len(self.lyrics) - len(instrument.notes)) > min(len(self.lyrics), len(instrument.notes)):
				continue			
			elif matches > max_matches:
				max_matches = matches
				min_offset = offset if min_offset is None or offset < min_offset else min_offset
				self.vocal_instrument = instrument
			elif matches == max_matches and (min_offset is None or offset < min_offset):
				min_offset = offset
				self.vocal_instrument = instrument

		if self.vocal_instrument is not None:
			self.is_instrumental = False
			self.other_instruments = [i for i in self.instruments if i != self.vocal_instrument]

			
	def _create_sung_lyrics(self):
		note_dict = OrderedDict([(n.start, n) for n in self.vocal_instrument.notes])
		for l_time, l in self.__lyrics_times.items():
			if not note_dict:
				break
			idx, distance = closest_match(list(note_dict.keys()), l_time)
			if distance <= self.lyrics_margin:
				key = list(note_dict.keys())[idx]
				self.lyrics_sung.append(Lyric_Sung(l.text, note_dict[key]))
				del note_dict[key]

	def _trim_overlapped_vocals(self):
		ls = self.lyrics_sung
		for i in range(len(ls) - 1):
			if ls[i].note.end > ls[i+1].note.start:
				ls[i].note.end = ls[i+1].note.start

	def _syllablify(self, phonemodel):
		lyrics_sung = []
		words = []
		last_syllable = []
		for l in self.lyrics_sung:
			ws = len(l.text.split())
			if ws > 1:
				duration_per = (l.note.end - l.note.start) / ws
				notes = [pretty_midi.Note(l.note.velocity, l.note.pitch, l.note.start + i * duration_per, l.note.start + (i + 1) * duration_per) for i in range(ws)]
				for i, (w, n) in enumerate(zip(l.text.rsplit(maxsplit=ws - 1), notes)):
					if i == 0:
						if len(w) - len(w.lstrip()):
							if last_syllable:
								words.append(last_syllable)
							words.append([Lyric_Sung(w, n)])
						else:
							last_syllable.append(Lyric_Sung(w, n))
							words.append(last_syllable)
					elif i < len(notes) - 1:
						words.append([Lyric_Sung(' ' + w, n)])
					else:
						last_syllable = [Lyric_Sung(' ' + w, n)]
			else:
				if len(l.text) - len(l.text.lstrip()):
					if last_syllable:
						words.append(last_syllable)
					last_syllable = [l]
				else:
					last_syllable.append(l)
		words.append(last_syllable)
		
		for w in words:
			subwords = [i.text for i in w]
			phonemes, syllables = get_phonemes(subwords, phonemodel)
			if syllables == 1:
				w[0].phonemes = [p[1] for p in phonemes if p[1] != '_'] if isinstance(phonemes[0], tuple) else phonemes
				w[0].text = ''.join(subwords)
				if w[0].phonemes and any(i.isdigit() for i in ''.join(w[0].phonemes)):
					lyrics_sung.append(w[0])
				for s in w[1:]:
					s.text = ''
					s.phonemes = ['_R_']
					lyrics_sung.append(s)
			else:
				split_p = phonemes_to_syllable_split(phonemes, syllables)

				syllable_text_phonemes = [[] for i in range(len(w))]
				word = ''.join(subwords)
				idx_c = 0
				for p in split_p:
					idx_p = 0
					text = ''
					phonemes = []
					if not p:
						continue
					while(idx_p) < len(p):
						i_syl = letter_to_syllable_idx(subwords, idx_c)
						if word[idx_c].isspace() or word[idx_c] == ',':
							text += word[idx_c]
							idx_c += 1
						else:
							letters = p[idx_p][0].replace('|', '')
							text += letters
							idx_c += len(letters)
							if p[idx_p][1] != '_':
								phonemes.append(p[idx_p][1])
							idx_p += 1
					syllable_text_phonemes[i_syl].append((text, phonemes))

				s_t_p = []
				empty = 0
				for i in syllable_text_phonemes:
					if i:
						s_t_p.append(i)
						for _ in range(empty):
							s_t_p.append([])
						empty = 0
					else:
						empty += 1

				for t_p, s in zip(s_t_p, w):
					if len(t_p) == 1:
						s.text = t_p[0][0]
						s.phonemes = t_p[0][1]
						if s.phonemes and any(i.isdigit() for i in ''.join(s.phonemes)):
							lyrics_sung.append(s)
					elif not t_p:
						s.text = ''
						s.phonemes = ['_R_']
						lyrics_sung.append(s)
					else:
						duration_per = (s.note.end - s.note.start) / len(t_p)
						notes = [pretty_midi.Note(s.note.velocity, s.note.pitch, s.note.start + i * duration_per, s.note.start + (i + 1) * duration_per) for i in range(len(t_p))]
						for i, n in zip(t_p, notes):
							if i[1] and any(j.isdigit() for j in ''.join(i[1])):
								lyrics_sung.append(Lyric_Sung(i[0], n, i[1]))

		self.lyrics_sung = lyrics_sung

	def instrumental_text_format(self, monophonic=False, include_velocity=True):
		note_events = []
		if self.key:
			instrumental_score = self.get_music21_score([i for i in self.other_instruments if not i.is_drum]).chordify()
			for e in [i for i in instrumental_score.secondsMap if i['element'].isClassOrSubclass(('Chord','Rest'))]:
				beat = e['element'].beat
				time = self.norm_time_to_tick(e['offsetSeconds'])
				if beat - int(beat) == 0:
					if beat == 1:
						note_events.append(
							Note_Event(time, None, True, '_DB_'))
					else:
						note_events.append(
							Note_Event(time, None, True, '_B_'))

				if e['element'].isChord:
					roman = music21.roman.romanNumeralFromChord(simplify_enharmonics_fast(e['element'], self.key), self.key).figure
					note_events.append(
						Note_Event(time, 'Chord', True, roman))
				else:
					note_events.append(
						Note_Event(time, None, True, '_REST_'))
			return text_format(note_events, include_velocity=False)
		else:		
			for instrument in self.other_instruments:
				ins_class = instrument_to_general_class(instrument)
				for note in instrument.notes:
					start = self.norm_time_to_tick(note.start)
					end = self.norm_time_to_tick(note.end)

					note_events.append(
						Note_Event(start, ins_class, True, note.pitch, velocity=note.velocity, will_end_at=end))
					note_events.append(
						Note_Event(end, ins_class, False, note.pitch))

			instrumental_note_events = sorted(note_events, key=lambda x: (x.time, -x.will_end_at))
			return text_format(instrumental_note_events, monophonic=monophonic, include_velocity=include_velocity)

	def vocal_text_format(self, include_velocity=True, include_phonemes=False, include_vowels=False, include_syllables=False):
		note_events = []
		for l in self.lyrics_sung:
			note = l.note
			text = l.text
			phonemes = l.phonemes
			start = self.norm_time_to_tick(note.start)
			end = self.norm_time_to_tick(note.end)
			if self.key:
				p = fix_pitch_outside_piano_range(note.pitch)
				chord = simplify_enharmonics_fast(music21.chord.Chord([p]), self.key)
				pitch = music21.roman.romanNumeralFromChord(chord, self.key).figure
				pitch += '{}'.format((p - self.key.tonic.midi)//12 + 4)
			else:
				pitch = note.pitch

			note_events.append(
				Note_Event(start, 'Vocal', True, pitch, velocity=note.velocity, lyric=text, phonemes=phonemes, will_end_at=end))
			note_events.append(
				Note_Event(end, 'Vocal', False, pitch, lyric=text, phonemes=phonemes))

		vocal_note_events = sorted(note_events, key=lambda x: (x.time, -x.will_end_at))

		return text_format(vocal_note_events, include_velocity=include_velocity, include_phonemes=include_phonemes, include_vowels=include_vowels, include_syllables=include_syllables)

	def fluidsynthesize(self, instruments=True, vocals=True, fs=44100, sf2_path=None, save=None):
		all_instruments = []
		if instruments:
			all_instruments += self.other_instruments
		if vocals:
			all_instruments.append(self.vocal_instrument)
		if not all_instruments:
			return "Nothing to do"
		waveforms = [i.fluidsynth(fs=fs, sf2_path=sf2_path) for i in all_instruments]
		synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
		for waveform in waveforms:
			synthesized[:waveform.shape[0]] += waveform
		synthesized /= np.abs(synthesized).max()
		write(save if save else "{}-{}.wav".format('instruments&vocals' if instruments and vocals else 'instruments' if instruments else 'vocals',
				os.path.split(self.midi_file)[-1].split('.')[0]), fs, synthesized)

	def text_events_tick_to_time(self, event_list, return_also_lyrics, roman_numerals):
		events_with_time = []
		for e in event_list:
			if e[:2] == 'W_':
				events_with_time.append('W_{}'.format(self.norm_tick_to_time(int(e.split('_')[1]))))
			elif roman_numerals and e[:3] == 'ON_':
				assert self.key
				pitch = music21.roman.RomanNumeral(e[3:-1], self.key).root().midi
				octave = int(e[-1]) - self.key.tonic.midi//12 + 1
				pitch += octave*12
				events_with_time.append('ON_{}'.format(pitch))
			else:
				events_with_time.append(e)
		if not return_also_lyrics:
			return events_with_time
		lyrics = []
		for e in event_list:
			if not '_' in e:
				if lyrics and lyrics[-1].isupper():
					lyrics[-1] += ' ' + e
				else:
					lyrics.append(e)
			elif e == '_C_':
				lyrics.append(',')
			elif e == 'N_DL':
				lyrics.append('\n\n')
			elif e == 'N_L':
				lyrics.append('\n')
			elif e == 'N_W':
				lyrics.append(' ')
		return events_with_time, lyrics

	def get_music21_score(self, instruments=None):
		if instruments is not None:
			original_instruments = self.instruments
			self.instruments = instruments
		with tempfile.TemporaryFile() as fp:
			self.write(fp)
			fp.seek(0)
			score = music21.midi.translate.midiStringToStream(fp.read())
		if instruments is not None:
			self.instruments = original_instruments
		return score



class Lyric_Sung(object):
	def __init__(self, text, note, phonemes=[]):
		self.text = text
		self.note = note
		self.phonemes = phonemes

	def __repr__(self):
		return 'Lyric_Sung(text="{}" phonemes="{}"" note={})'.format(
			repr(self.text), repr(self.phonemes), repr(self.note))

	def __str__(self):
		return '"{} - {}" sung as: {}'.format(self.text, ' '.join(self.phonemes) if self.phonemes else 'no phonemes', self.note)


class Note_Event(object):
	def __init__(self, time, instrument_class, note_on, pitch, velocity=None, lyric=None, phonemes=[], will_end_at=pretty_midi.MAX_TICK):
		self.time = time
		self.instrument_class = instrument_class
		self.note_on = note_on
		self.pitch = pitch
		self.velocity = velocity
		self.lyric = lyric
		self.phonemes = phonemes
		self.will_end_at = will_end_at

	def __repr__(self):
		return 'Note_Event(time (in ticks): {} instrument: {} {} pitch: {} velocity: {} lyric:{})'.format(
			repr(self.time), repr(self.instrument_class), 'note_on' if self.note_on else 'note_off', repr(self.pitch), repr(self.velocity), repr(self.lyric), repr(self.phonemes))

	def __str__(self):
		return '"{}"_"{}"_{}_{}{}'.format(self.lyric if self.lyric else self.instrument_class, ' '.join(self.phonemes) if self.phonemes else ' ',
			'ON' if self.note_on else 'OFF', self.pitch, '_' + str(self.velocity) if self.note_on else '')


def merge_lyrics_with_vocal_events(lyrics, text_events, phonemodel='cmudict-20170708.o8.fst'):
	phonemodel = Phonetisaurus(phonemodel)
	phonemes_list, syllables_list = zip(*[get_phonemes(l, phonemodel) for l in lyrics.strip('<|endoftext|>').split()])
	lyrics_events = []
	for phonemes, syllables in zip(phonemes_list, syllables_list):
		if syllables == 1:
			if isinstance(phonemes[0], tuple):
				phonemes = [p[1] for p in phonemes]
			lyrics_events.append(' '.join([p for p in phonemes if p != '_']))
		else:
			split_p = phonemes_to_syllable_split(phonemes, syllables)
			for p in split_p:
				lyrics_events.append(' '.join(i[1] for i in p if i[1] != '_'))

	merged_events = []
	for t in text_events:
		if t[:3] == 'ON_' and merged_events and merged_events[-1] != '_R_':
			if lyrics_events:
				merged_events.append(lyrics_events.pop(0))
			else:
				break
		merged_events.append(t)

	return merged_events

def simplify_enharmonics_fast(chord, key):
	chord.closedPosition(inPlace=True)
	key_pitches = [p.name for p in key.pitches]

	new_pitches = []
	for pitch in chord.pitches:
		if pitch.name in key_pitches:
			new_pitches.append(pitch)
		else:
			candidates = pitch.getAllCommonEnharmonics()
			found = False
			for p in candidates:
				if p.name in key_pitches:
					new_pitches.append(p)
					found = True
					break
			if not found:
				return simplify_enharmonics_slow(chord, key)
	chord.pitches = new_pitches
	return chord

def simplify_enharmonics_slow(chord, key):
	newPitches = [key.asKey('major').tonic]
	for oldPitch in chord.pitches:
		candidates = [oldPitch] + oldPitch.getAllCommonEnharmonics()
		newPitch = min(candidates, key=lambda x: music21.pitch._dissonanceScore(newPitches + [x]))
		newPitches.append(newPitch)
	chord.pitches = newPitches[1:]
	return chord