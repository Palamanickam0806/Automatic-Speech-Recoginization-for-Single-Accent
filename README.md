# Automatic-Speech-Recoginization-for-Single-Accent

# Demo
Run the below command in this directory to print the output. <br />
 ```python test.py```
 
In ```test.py``` directly change the audio file which exist in test audio folder as this model work fine for similar accent.

Model download Drive link : [ASR - MODEL](https://drive.google.com/drive/folders/1skKYg9oaCQ0N0T2VXLmV5dS7h7Bu9vMN?usp=sharing)

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
  * Parameter size of 10M.
 # 🧠 ASR Model Architecture

| Layer (type)        | Input Shape           | Output Shape          | # Params      | Trainable |
|---------------------|------------------------|------------------------|---------------|-----------|
| **ASRModel**        | `[1, 128, 486]`        | `[1, 120, 28]`         | —             | ✅         |
| ├── Conv2d          | `[1, 1, 128, 486]`     | `[1, 32, 128, 486]`    | 320           | ✅         |
| ├── BatchNorm2d     | `[1, 32, 128, 486]`    | `[1, 32, 128, 486]`    | 64            | ✅         |
| ├── GELU            | `[1, 32, 128, 486]`    | `[1, 32, 128, 486]`    | —             | —         |
| ├── Conv2d          | `[1, 32, 128, 486]`    | `[1, 64, 128, 486]`    | 18,496        | ✅         |
| ├── BatchNorm2d     | `[1, 64, 128, 486]`    | `[1, 64, 128, 486]`    | 128           | ✅         |
| ├── GELU            | `[1, 64, 128, 486]`    | `[1, 64, 128, 486]`    | —             | —         |
| **→ Flattened to**  | `[1, 8192, 486]`       |                        |               |           |
| ├── Conv1d          | `[1, 8192, 486]`       | `[1, 256, 486]`        | 6,291,712     | ✅         |
| ├── BatchNorm1d     | `[1, 256, 486]`        | `[1, 256, 486]`        | 512           | ✅         |
| ├── GELU            | `[1, 256, 486]`        | `[1, 256, 486]`        | —             | —         |
| ├── GRU             | `[120, 256]`           | `[120, 512]`           | 4,337,664     | ✅         |
| ├── LayerNorm       | `[1, 120, 512]`        | `[1, 120, 512]`        | 1,024         | ✅         |
| ├── Linear          | `[1, 120, 512]`        | `[1, 120, 256]`        | 131,328       | ✅         |
| ├── GELU            | `[1, 120, 256]`        | `[1, 120, 256]`        | —             | —         |
| ├── Dropout         | `[1, 120, 256]`        | `[1, 120, 256]`        | —             | —         |
| ├── Linear          | `[1, 120, 256]`        | `[1, 120, 28]`         | 7,196         | ✅         |

---

 # 📊 Model Summary

- **Total Parameters**: `10,788,444`
- **Trainable Parameters**: `10,788,444`
- **Model Size (Parameters)**: ~43.15 MB
- **Forward/Backward Memory Usage**: ~98.80 MB
- **Input Size**: ~0.25 MB

 ## Language Model
   * It’s a statistical model that is designed to analyze the pattern of human language and predict the likelihood of a sequence of words or tokens.
   * To tackled spelling errors in predictions  a novel static language model to added at post processing , where the incorrect word is replaced with best matching and ngram sequence word by checking in arpa file ```3gram.arpa```  


# Results
  **Character Error Rate (CER)** = 0.065 <br />
  **Word Error Rate (WER)** = 0.1478 <br />
  The ```prediction vs reference of test data.csv``` have prediction and references of each test audio file by testing through our ASR model 

# To be Done: 
  * QUANTIZATION
      Converts high-precision (FP32) weights to lower precision (FP16 or INT8) for smaller memory usage and faster execution. <br />
      Reduces inference latency on hardware like TPUs, edge devices. <br />
      INT8 is ideal for real-time applications, while FP16 balances speed and accuracy.use significantly less energy, which is crucial for battery-powered and IoT devices. 
  * APPROXIMATION
      Reduces redundant computations in convolution layers using techniques like depthwise separable convolutions. <br />
## Contact : [email](valliaravind@gmail.com)
      Uses simplified activation functions (e.g., replacing expensive functions like exp with piecewise linear approximations

