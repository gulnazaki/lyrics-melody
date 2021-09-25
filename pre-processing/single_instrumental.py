import load_midi
import argparse

def get_events(file, include_velocity, norm_resolution, music_analysis):
	midi = load_midi.Midi(file, norm_resolution=norm_resolution, music_analysis=music_analysis)
	return midi.instrumental_text_format(include_velocity=include_velocity)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create a newline separated instrumental txt for a single MIDI file')
	
	parser.add_argument('--input', '-i', type=str, help='Path of midi instrumental', required=True)
	parser.add_argument('--output', '-o', type=str, default='instrumental', help='Base filename to save the output txt')
	parser.add_argument('--include-velocity', '-iv', action='store_true', help='Include velocity in the dataset')
	parser.add_argument('--norm-resolution', '-nr', type=int, help='Normalize by changing resolution of midi files')
	parser.add_argument('--music-analysis', '-ma', action='store_true', help='Do music analysis to extract chords, etc.')

	args = parser.parse_args()
	
	events = get_events(file=args.input,
    	 				include_velocity=args.include_velocity,
		 				norm_resolution=args.norm_resolution,
		 				music_analysis=args.music_analysis)

	with open(args.output + '.txt', 'w') as f:
		f.write('\n'.join(events))
		