# Lyrics and Vocal Melody Generation conditioned on Accompaniment

## About
This repo is provided as a supplement to the paper **Lyrics and Vocal Melody Generation conditioned on Accompaniment**.
It has been currently accepted at the 2nd Workshop on NLP for Music and Spoken Audio [NLP4MuSA 2021](https://sites.google.com/view/nlp4musa-2021).

It includes:
- All code to recreate the results of the paper
- The full instrumental dataset `dataset.parquet` and the reduced chord dataset `dataset_chords.parquet`, as well as their token vocabularies
- A few generated samples and their corresponding lyrics


*This work is the continuation of my [Master's thesis](http://artemis.cslab.ece.ntua.gr:8080/jspui/handle/123456789/17907) (MEng in ECE at [NTUA](https://www.ece.ntua.gr/en))*

## Code

The provided code includes scripts to:
- pre-process MIDI tracks and bring them to the used text event format
- build the full and chord reduced dataset (as well as the token vocabularies) from the LMD matched dataset
- train the vanilla encoder-decoder architecture
- train the decoupled architecture
- warm-start and finetune the lyrics language model
- adjust Mellotron in order to use our text event format
- train all models and generate samples on Colab (Jupyter Notebooks)
