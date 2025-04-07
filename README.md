# Automatic-Speech-Recoginization-for-Single-Accent

# Working
  * Spectrogram generator that converts raw audio to spectrograms.
  * Acoustic model that takes the spectrograms as input and outputs a matrix of probabilities over characters over time.The neural network is present in this acoustic model
  * CTC Decoder (coupled with a language model) that generates possible sentences from the probability matrix.
