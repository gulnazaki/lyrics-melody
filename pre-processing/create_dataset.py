import argparse
import load_midi
import os
from polyglot.detect import Detector
import json
import jsonlines
import csv
import pandas as pd
from multiprocessing import Process

def valid_lyrics(lyrics):
	t = "".join(lyrics)
	if not t.isascii() or not is_english(t):
		return False
	else:
		return True

def is_english(text):
	try:
		det = Detector(text)
		lang = det.language.name
		reliable = det.reliable
	except:
		return False
	if lang != "English" or not reliable:
		return False
	return True


class MIDIStream(list):
	def __init__(self, lmd_path, track_matches, **kwargs):
		self.lmd_path = lmd_path
		self.track_matches = track_matches
		self.include_filenames = kwargs['include_filenames']
		self.include_lyrics = kwargs['include_lyrics']
		self.instrumental_type = kwargs['instrumental_type']
		self.include_velocity = kwargs['include_velocity']
		self.include_phonemes = kwargs['include_phonemes']
		self.include_vowels = kwargs['include_vowels']
		self.include_syllables = kwargs['include_syllables']
		self.norm_resolution = kwargs['norm_resolution']
		self.music_analysis = kwargs['music_analysis']

	def __iter__(self):
		for match in self.track_matches:
			data = self.select_midi(match, self.include_velocity, self.include_phonemes, self.include_vowels, self.include_syllables, self.norm_resolution, self.music_analysis, self.instrumental_type >= 2)
			if data is not None:
				data_dict = {}
				if self.include_filenames:
					data_dict['file'] = data[0]
				if self.include_lyrics:
					data_dict['lyrics'] = json.dumps(data[1])					
				if self.instrumental_type % 2 == 1:
					data_dict['instrumental'] = json.dumps(data[2])
				if self.instrumental_type >= 2:
					data_dict['monophonic'] = json.dumps(data[3])
				data_dict['vocal'] = json.dumps(data[4])
				yield (data_dict)

	def select_midi(self, match, include_velocity, include_phonemes, include_vowels, include_syllables, norm_resolution, music_analysis, include_monophonic=False):
		if isinstance(match, tuple):
			track, midis = match
			trackpath = os.path.join(track[2], track[3], track[4], track)
		else:
			trackpath, midi = os.path.split(match)
			midis = [os.path.splitext(midi)[0]]

		for m in midis:
			midipath = os.path.join(trackpath, m) + ".mid"
			try:
				midi = load_midi.Midi(os.path.join(self.lmd_path, midipath), norm_resolution=norm_resolution, music_analysis=music_analysis)
			except KeyboardInterrupt:
				exit()
			except Exception as e:
				print(e, midipath)
				continue

			if midi.is_instrumental:
				continue
			else:
				lyrics = [l.text for l in midi.lyrics_sung]
				if not valid_lyrics(lyrics):
					continue
				else:
					return (midipath,
							lyrics,
							midi.instrumental_text_format(include_velocity=include_velocity),
							midi.instrumental_text_format(monophonic=True, include_velocity=include_velocity) if include_monophonic else [],
							midi.vocal_text_format(include_velocity=include_velocity,
												   include_phonemes=include_phonemes,
												   include_vowels=include_vowels,
												   include_syllables=include_syllables))
		return None

	def __len__(self):
		return 1


class DataStream(list):
	def __init__(self, tmp_outputs):
		self.tmp_outputs = tmp_outputs

	def __iter__(self):
		for tmp_output in self.tmp_outputs:
			with jsonlines.open(tmp_output) as reader:
				for line in reader:
					yield line

	def __len__(self):
		return 1


def load(lmd_path, chunk, tmp_output, **kwargs):
	midi_stream = MIDIStream(lmd_path, chunk, **kwargs)
	with jsonlines.open(tmp_output, 'w') as writer:
		writer.write_all(midi_stream)

def multiprocessing_load(lmd_path, match_scores_path, output, workers, parquet_for_selected, convert_to_parquet, **kwargs):
	if parquet_for_selected:
		try:
			df = pd.read_parquet(parquet_for_selected, columns=["file"])
		except ValueError:
			exit("The provided parquet doesn't have a 'file' column")
		except:
			exit("Error with provided parquet")
		sorted_matches = list(df['file'])
	else:
		with open(match_scores_path, 'r') as f:
			track_matches = json.load(f)
			sorted_matches = [(track, sorted(midis.keys(), key=lambda s: midis[s], reverse=True)) for track, midis in track_matches.items()]

	track_chunks = [sorted_matches[i::workers] for i in range(workers)]

	jobs = []
	tmp_outputs = []
	for i, chunk in enumerate(track_chunks):
		tmp_output = "{}-{}.jsonl".format(output, i)
		j = Process(target=load, args=(lmd_path, chunk, tmp_output), kwargs=kwargs)
		jobs.append(j)
		tmp_outputs.append(tmp_output)
	
	for j in jobs:
		j.start()

	for j in jobs:
		j.join()

	data_stream = DataStream(tmp_outputs)

	csv_columns = []
	if kwargs['include_filenames']:
		csv_columns.append('file')
	if kwargs['include_lyrics']:
		csv_columns.append('lyrics')
	if kwargs['instrumental_type'] % 2 == 1:
		csv_columns.append('instrumental')
	if kwargs['instrumental_type'] >= 2:
		csv_columns.append('monophonic')
	csv_columns.append('vocal')

	with open("{}.csv".format(output), 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
		writer.writeheader()
		writer.writerows(data_stream)
	
	if convert_to_parquet:
		df = pd.read_csv("{}.csv".format(output))
		df.to_parquet("{}.parquet".format(output))
		os.remove("{}.csv".format(output))

	for tmp_output in tmp_outputs:
		os.remove(tmp_output)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create a single dataset csv that contains: selected midi, lyrics, instrumental and vocal events \
										for each (non instrumental and valid) MSD track found in LMD (matched or aligned)')
	
	parser.add_argument('--lmd-path', '-lmd', type=str, help='Path of lmd_matched or lmd_aligned directory', required=True)
	parser.add_argument('--match-scores-path', '-ms', type=str, help='Path of the match_scores.json')
	parser.add_argument('--output', '-o', type=str, default='dataset', help='Base filename to save the temporary jsonl files and the data csv')
	parser.add_argument('--workers', '-n', default=4, type=int, help='Number of processes to spawn')
	parser.add_argument('--parquet-for-selected', '-pfs', type=str, help='Path of a parquet file that contains the already selected midis, overrides match scores')
	parser.add_argument('--no-filenames', '-nf', action='store_true', help="Don't include filenames in the output")
	parser.add_argument('--no-lyrics', '-nl', action='store_true', help="Don't include lyrics in the output")
	parser.add_argument('--instrumental-type', '-it', default=1, type=int, help="Polyphonic instrumental format : 1, monophonic : 2, both : 3")
	parser.add_argument('--convert-to-parquet', '-cpt', action='store_true', help="Convert to parquet for faster reading and smaller size")
	parser.add_argument('--include-velocity', '-iv', action='store_true', help='Include velocity in the dataset')
	parser.add_argument('--include-phonemes', '-iphon', action='store_true', help='Include phonemes in the dataset')
	parser.add_argument('--include-vowels', '-ivow', action='store_true', help='Include vowels in the dataset')
	parser.add_argument('--include-syllables', '-isyl', action='store_true', help='Include syllables in the dataset')
	parser.add_argument('--norm-resolution', '-nr', type=int, help='Normalize by changing resolution of midi files')
	parser.add_argument('--music-analysis', '-ma', action='store_true', help='Do music analysis to extract chords, etc.')

	args = parser.parse_args()
	
	if not args.match_scores_path and not args.parquet_for_selected:
		exit("Provide match scores or parquet with selected midis")
	
	multiprocessing_load(lmd_path=args.lmd_path,
		 				 match_scores_path=args.match_scores_path,
		 				 output=args.output,
		 				 workers=args.workers,
		 				 parquet_for_selected=args.parquet_for_selected,
		 				 convert_to_parquet=args.convert_to_parquet,
		 				 include_filenames=not args.no_filenames,
		 				 include_lyrics=not args.no_lyrics,
		 				 instrumental_type=args.instrumental_type,
		 				 include_velocity=args.include_velocity,
		 				 include_phonemes=args.include_phonemes,
		 				 include_vowels=args.include_vowels,
		 				 include_syllables=args.include_syllables,
		 				 norm_resolution=args.norm_resolution,
		 				 music_analysis=args.music_analysis)