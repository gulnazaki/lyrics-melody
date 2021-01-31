from statistics import mean
from bisect import bisect_left
import re

MAX_WAIT = 1000
VEL_QUANT = 4
MIN_PITCH = 21
MAX_PITCH = 108
LYRICS_THRESH = 50
VALID_THRESH = 0.6
MAX_CHARS = 16
MAX_WORDS = 4
INSTRUMENTS = ["PIANO", "GUITAR", "BASS", "STRINGS", "WIND", "SYNTH", "DRUMS", "EFFECTS"]

note_on_dict = {(instrument, pitch) : -1 for instrument in ["Vocal"] + INSTRUMENTS for pitch in list(range(MIN_PITCH, MAX_PITCH + 1)) + [0]}

windows_chars = {
	'\x82' : '',
	'\x84' : '',
	'\x85' : '',
	'\x88' : '',
	'\x91' : "'",
	'\x92' : "'",
	'´'	   : "'",
	'\x93' : '',
	'\x94' : '',
	'\x95' : ' ',
	'\x96' : '-',
	'\x97' : '',
	'\x99' : ' ',
	'\xa0' : ' ',
	'\xa6' : '',
	'\xab' : '',
	'\xbb' : '',
	'\xbc' : '',
	'\xbd' : '',
	'\xbe' : '',
	'\xbf' : '',
	'\xa8' : '',
	'\xb1' : '' 
}

windows_regex = re.compile('(' + '|'.join(windows_chars.keys()) + ')')
clean_regex = re.compile(r"-|[^A-Za-z \n']+")
space_regex = re.compile(r"\s+")

def lyrics_margin(tempi):
	if tempi.size != 0:
		return 16/mean(tempi)
	return 0.16

def fix_special_chars(objects):
	for obj in objects:
		if '\x00' in obj.text:
			obj.text = obj.text.split('\x00', 1)[0]
		obj.text = windows_regex.sub(lambda match: windows_chars[match.group(0)], obj.text)

def valid_lyric(time, text):
	return time != 0 and not any(t in text.lower() for t in ['karaoke', 'www.', '.com', 'http', 'instrumental']) \
					 and not all(t in text for t in ['(', ')'])

def clear_text(lyrics, upper_text):
	clean_lyrics = []
	for l in lyrics:
		c_text = clean_regex.sub(lambda match: " " if match.group(0) == "-" else "", l.text.upper() if upper_text else l.text)
		c_text = space_regex.sub(" ", c_text)
		l.text = c_text.rstrip()
		if l.text and len(l.text) <= MAX_CHARS and len(l.text.split()) <= MAX_WORDS :
			clean_lyrics.append(l)
	return clean_lyrics

def closest_match(notes, lyric):
	idx = bisect_left(notes, lyric)
	if idx == 0:
		return idx, (notes[0] - lyric)
	elif idx == len(notes):
		return (idx - 1), (lyric - notes[-1])
	else:
		left = lyric - notes[idx - 1]
		right = notes[idx] - lyric
		
		if left < right:
			return (idx - 1), left
		else:
			return idx, right

def match_lyrics(lyrics, notes, margin):
	offset = 0
	matches = 0
	for l in lyrics:
		if not notes:
			break
		elif l in notes:
			matches += 1
			notes.remove(l)
		else:
			idx, distance = closest_match(notes, l)
			offset += distance
			if distance <= margin:
				matches += 1
			notes.pop(idx)
	return offset, matches

def remove_instruments(instruments, events):
	print(len(events))
	without_instruments = [event for event in events if not any(i in event for i in instruments)]
	print(len(without_instruments))
	clean_text = []
	wait = 0
	for event in without_instruments:
		if 'W_' in event:
			wait += int(event.split('_')[1])
			if wait >= MAX_WAIT:
				clean_text.append('W_{}'.format(MAX_WAIT))
				wait -= MAX_WAIT
		else:
			if wait > 0:
				clean_text.append('W_{}'.format(wait))
				wait = 0
			clean_text.append(event)

	return clean_text

def instrument_to_general_class(instrument):
	if instrument.is_drum:
		return "DRUMS"	
	program = instrument.program
	if program < 0 or program > 127:
		raise ValueError('Invalid program number {}, should be between 0 and'
						 ' 127'.format(program_number))
	elif program < 24 or program == 108:
		return "PIANO"
	elif program < 32 or program == 104 or program == 107:
		return "GUITAR"
	elif program < 40 or program == 105 or program == 106:
		return "BASS"
	elif program < 56 or program == 110:
		return "STRINGS"
	elif program < 80 or program == 109 or program == 111:
		return "WIND"
	elif program < 104:
		return "SYNTH"
	elif program < 120:
		return "DRUMS"
	else:
		return "EFFECTS"

def fix_pitch_outside_piano_range(pitch):
	while pitch > MAX_PITCH:
		pitch -= 12
	while pitch < MIN_PITCH:
		pitch += 12
	return pitch

def valid_event(note_on, instrument, pitch, time, will_end_at, monophonic):
	if monophonic and instrument != "DRUMS":
		key = (instrument, 0)
	else:
		key = (instrument, pitch)
	note_ends = note_on_dict[key]
	if note_on and note_ends < 0:
		note_on_dict[key] = will_end_at
		return True
	elif not note_on and note_ends == time:
		note_on_dict[key] = -1
		return True
	else:
		return False

def text_format(events, monophonic=False, include_velocity=True):
	last_note_tick = 0
	last_lyric = None
	event_text_format = []

	for note_event in events:
		time = note_event.time
		tick_diff = time - last_note_tick
		note_on = note_event.note_on
		instrument = note_event.instrument_class
		pitch = fix_pitch_outside_piano_range(note_event.pitch)
		lyric = note_event.lyric

		if not valid_event(note_on, instrument, pitch, time, note_event.will_end_at, monophonic=monophonic):
			continue

		if tick_diff != 0:
			while tick_diff > MAX_WAIT:
				event_text_format.append('W_{}'.format(MAX_WAIT))
				tick_diff -= MAX_WAIT
			event_text_format.append('W_{}'.format(tick_diff))

		if lyric is not None and lyric != last_lyric:
			event_text_format.append(lyric)
			last_lyric = lyric
		
		if note_on:
			velocity = '_V{}'.format((note_event.velocity // VEL_QUANT) * VEL_QUANT + (VEL_QUANT >> 1)) if include_velocity else ''
			event_text_format.append('{}ON_{}{}'.format(instrument + '_' if instrument != 'Vocal' else '', pitch, velocity))
		else:
			if instrument == 'Vocal':
				event_text_format.append('_OFF_')
			else:
				event_text_format.append('{}_OFF_{}'.format(instrument, pitch))			
		
		last_note_tick = note_event.time

	return event_text_format
