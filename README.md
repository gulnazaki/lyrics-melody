# thesis

## About
This repo contains code I have written for my Master's thesis (MEng in ECE at NTUA).

The subject is **Lyrics and Vocal Melody Generation, conditioned on Accompaniment**.

### LMD Tranformer
For the first part I experiment with a [Performer](https://arxiv.org/abs/2009.14794), 
a Tranformer-derivative Encoder-Decoder architecture that can efficiently model long sequences,
on MIDI Data. I used the [Lakh MIDI Dataset](https://arxiv.org/abs/2009.14794), which I further 
processed to decouple instrumental and vocal events and provide a text event representation
(similar to the one used in [MuseNet](https://openai.com/blog/musenet/)) as training data.