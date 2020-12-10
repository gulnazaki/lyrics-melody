import pretty_midi
from utils import *
from heapq import merge
from collections import OrderedDict

class Midi(pretty_midi.PrettyMIDI):
	def __init__(self, midi_file=None, resolution=220, initial_tempo=120.):
		super().__init__(midi_file, resolution, initial_tempo)
		
		if midi_file is not None:
			self.lyrics_margin = lyrics_margin(self.get_tempo_changes()[1])
			self._reclassify_lyrics_and_text()
			self._embed_whitespaces()
			self.lyrics = clear_and_upper(self.lyrics)
			self._sort_notes_and_lyrics()
			
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


	def _reclassify_lyrics_and_text(self):
		fix_special_chars(self.lyrics)
		fix_special_chars(self.text_events)
		valid_lyrics = [l for l in self.lyrics if valid_lyric(l.time, l.text)]
		valid_text_events = [t for t in self.text_events if valid_lyric(t.time, t.text)]

		if len(valid_lyrics) > len(valid_text_events):
			self.lyrics = valid_lyrics
		else:
			self.lyrics = [pretty_midi.Lyric(t.text, t.time) for t in valid_text_events]


	def _embed_whitespaces(self):
		lyrics = []
		for l in self.lyrics:
			if not l.text.isspace() and l.text != '':
				if l.text[0] == '\\':
					l.text = l.text[1:]
					if lyrics:
						lyrics[-1].text += '\n\n'
				elif l.text[0] == '/':
					l.text = l.text[1:]
					if lyrics:
						lyrics[-1].text += '\n'
				
				no_trailing = l.text.lstrip()
				trailing_spaces = len(l.text) - len(no_trailing)
				if trailing_spaces:
					l.text = no_trailing
					if lyrics:
						lyrics[-1].text += ' ' * trailing_spaces
			
				lyrics.append(l)
			else:
				if lyrics:
					lyrics[-1].text += '\n'
		
		self.lyrics = lyrics

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


	def instrumental_text_format(self, monophonic=False):
		note_events = []
		for instrument in self.other_instruments:
			ins_class = instrument_to_general_class(instrument)
			for note in instrument.notes:
				start = self.time_to_tick(note.start)
				end = self.time_to_tick(note.end)

				note_events.append(
					Note_Event(start, ins_class, True, note.pitch, velocity=note.velocity, will_end_at=end))
				note_events.append(
					Note_Event(end, ins_class, False, note.pitch))

		instrumental_note_events = sorted(note_events, key=lambda x: (x.time, -x.will_end_at))

		return text_format(instrumental_note_events, monophonic=monophonic)


	def vocal_text_format(self, max_size=None):
		note_events = []
		for l in self.lyrics_sung:
			note = l.note
			text = l.text
			start = self.time_to_tick(note.start)
			end = self.time_to_tick(note.end)

			note_events.append(
				Note_Event(self.time_to_tick(note.start), 'Vocal', True, note.pitch, velocity=note.velocity, lyric=text, will_end_at=end))
			note_events.append(
				Note_Event(self.time_to_tick(note.end), 'Vocal', False, note.pitch, lyric=text))

		vocal_note_events = sorted(note_events, key=lambda x: (x.time, -x.will_end_at))

		return text_format(vocal_note_events)


class Lyric_Sung(object):
	def __init__(self, text, note):
		self.text = text
		self.note = note

	def __repr__(self):
		return 'Lyric_Sung(text="{}" note={})'.format(
			self.text, repr(self.note))

	def __str__(self):
		return '"{}" sung as: {}'.format(self.text, self.note)


class Note_Event(object):
	def __init__(self, time, instrument_class, note_on, pitch, velocity=None, lyric=None, will_end_at=pretty_midi.MAX_TICK):
		self.time = time
		self.instrument_class = instrument_class
		self.note_on = note_on
		self.pitch = pitch
		self.velocity = velocity
		self.lyric = lyric
		self.will_end_at = will_end_at

	def __repr__(self):
		return 'Note_Event(time (in ticks): {} instrument: {} {} pitch: {} velocity: {} lyric:{})'.format(
			self.time, self.instrument_class, 'note_on' if self.note_on else 'note_off', self.pitch, self.velocity, self.lyric)

	def __str__(self):
		return '"{}"_{}_{}{}'.format(self.lyric if self.lyric is not None else self.instrument_class,
			'ON' if self.note_on else 'OFF', self.pitch, '_' + str(self.velocity) if self.note_on else '')
