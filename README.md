# Automatic-Speech-Recoginization-for-Single-Accent

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

