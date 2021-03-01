import load_midi
import pandas as pd
import os
import argparse
import json
from collections import Counter

def create_vocabs(_type_, output_prefix, dataset=None, thresh=1, include_velocity=False, include_vowels=False, monophonic=True, for_decoupled=False, music_analysis=False):
	special_tokens = ['<pad>', '<bos>', '<eos>']

	instruments = [ins + "_" for ins in load_midi.INSTRUMENTS] if _type_ == 'instrumental' else [""]

	waits = ['W_{}'.format(w) for w in range(1, load_midi.MAX_WAIT + 1)]
	
	if music_analysis:
		ext = os.path.splitext(dataset)[1]
		if ext != '.csv' and ext != ".parquet":
			exit("Provide a .csv or .parquet file for vocal vocabulary")	
		try:
			df = pd.read_csv(dataset, usecols=[_type_]) if ext == '.csv' else pd.read_parquet(dataset, columns=[_type_])
		except ValueError:
			exit("The provided file doesn't have a {} column".format(_type_))
		except:
			exit("Problem with dataset file")
		if _type_ == 'instrumental':
			noteons = ['_DB_', '_B_']
			noteoffs = ['_REST_']
		else:
			noteons = []
			noteoffs = ['_OFF_']
		note_cnt = Counter([event for f in df[_type_] for event in json.loads(f) if event[:3] == 'ON_'])
		noteons += [tok for tok, cnt in note_cnt.most_common() if cnt >= thresh]
	else:
		if include_velocity:
			noteons = ['{}ON_{}_V{}'.format(instrument, pitch, velocity) for instrument in instruments
																			for pitch in range(load_midi.MIN_PITCH, load_midi.MAX_PITCH + 1)
																				for velocity in range(load_midi.VEL_QUANT >> 1, 128, load_midi.VEL_QUANT)]
		else:
			noteons = ['{}ON_{}'.format(instrument, pitch) for instrument in instruments
																for pitch in range(load_midi.MIN_PITCH, load_midi.MAX_PITCH + 1)]
		
		if monophonic:
			if _type_ == 'instrumental':
				noteoffs = ['{}OFF'.format(instrument) for instrument in instruments]
			else:
				noteoffs = ['_OFF_']
		else:
			noteoffs = ['{}OFF_{}'.format(instrument, pitch) for instrument in instruments
														 	for pitch in range(load_midi.MIN_PITCH, load_midi.MAX_PITCH + 1)] 

	if _type_ == 'vocal':
		singing_tokens = ['N_DL', 'N_L', 'N_W', '_C_', '_R_']
		if for_decoupled:
			if include_vowels:
				singing_tokens += [v + stress for v in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'] for stress in ['0', '1', '2']]
		else:
			if not music_analysis:
				ext = os.path.splitext(dataset)[1]
				if ext != '.csv' and ext != ".parquet":
					exit("Provide a .csv or .parquet file for vocal vocabulary")	
				try:
					df = pd.read_csv(dataset, usecols=["vocal"]) if ext == '.csv' else pd.read_parquet(dataset, columns=["vocal"])
				except ValueError:
					exit("The provided file doesn't have a 'vocals' column")
				except:
					exit("Problem with dataset file")

			singing_cnt = Counter([event for f in df['vocal'] for event in json.loads(f) if '_' not in event])
			singing_tokens += [tok for tok, cnt in singing_cnt.most_common() if cnt >= thresh]
	else:
		singing_tokens = []

	if output_prefix:
		output_prefix += '_'
	with open('{}{}.vocab'.format(output_prefix, _type_), 'w') as f:
		f.write("\n".join(special_tokens + waits + noteons + noteoffs + singing_tokens) + "\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create vocal and instrumental (monophonic or polyphonic) vocabularies. Every possible music event is included, \
													singing tokens require a dataset csv or parquet file.')
	
	parser.add_argument('--dataset-file', '-df', type=str, help='Path of a valid dataset csv or parquet file.')
	parser.add_argument('--threshold', '-t', type=int, default= 1, help='A singing token has to appear at least this number of times to be included')
	parser.add_argument('--output-prefix', '-o', type=str, default='', help='Prefix of the output files: <out>_instrumental.vocab and <out>_vocal.vocab')
	parser.add_argument('--include-velocity', '-iv', action='store_true', help='Include velocity in the vocabularies')
	parser.add_argument('--include-vowels', '-ivow', action='store_true', help='Include vowels in the vocabularies')
	parser.add_argument('--monophonic', '-m', action='store_true', help='Create vocabulary for monophonic input (output is always monophonic)')
	parser.add_argument('--for-decoupled', '-fd', action='store_true', help='Create vocal vocabulary for decoupled performer')
	parser.add_argument('--music-analysis', '-ma', action='store_true', help='Music analysis vocabulary with roman numerals and rests, etc.')

	args = parser.parse_args()

	create_vocabs(_type_='instrumental', output_prefix=args.output_prefix, dataset=args.dataset_file, thresh=args.threshold, include_velocity=args.include_velocity, monophonic=args.monophonic, music_analysis=args.music_analysis)
	create_vocabs(_type_='vocal', output_prefix=args.output_prefix, dataset=args.dataset_file, thresh=args.threshold, include_velocity=args.include_velocity, include_vowels=args.include_vowels, for_decoupled=args.for_decoupled, music_analysis=args.music_analysis)
