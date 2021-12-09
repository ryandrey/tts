#!/bin/bash

pip install -r tts/requirements.txt

wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
rm -rf LJSpeech-1.1.tar.bz2

git clone https://github.com/NVIDIA/waveglow.git

wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip -q alignments.zip

python3 tts/train.py -c tts/tts/config.json