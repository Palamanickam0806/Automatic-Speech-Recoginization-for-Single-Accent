# Automatic-Speech-Recoginization-for-Single-Accent

# Demo
Run the below command in this directory to print the output. <br />
 ```python test.py```
 
In ```test.py``` directly change the audio file which exist in test audio folder as this model work fine for similar accent.

For model file ping me [email](valliaravind@gmail.com)

# Dataset:  
 ### Link : [LJ Speech Data](https://keithito.com/LJ-Speech-Dataset/)  <br />
 This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. A transcription is provided for  each clip. Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.  
 The audio is extracted then augmented ( Raw and Specaugment both) and finally trained the model using 40 Hours of audio dataset
 
# Working
  * Spectrogram generator that converts raw audio to spectrograms.
  * ASR model that takes the spectrograms as input and outputs a matrix of probabilities over characters over time.The neural network is present in this ASR model.
  * CTC Decoder (coupled with a language model) that generates possible sentences from the probability matrix.

# Model
  * Feature extraction and padding
  * Encoder (2 CNN2d + 1 CNN1d + 5bi-GRU + FC)
  * CTC loss and forced alignment of character with the encoder output.
  * Inference CTC Decoder
  * Statistical Language Model for spelling correction and next word prediction.

# Results
  ## Character Error Rate (CER) = 0.065
  ## Word Error Rate (WER) = 0.1478
