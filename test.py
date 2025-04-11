import torch 
import librosa 
import numpy as np 
import heapq
import difflib

char_map = {1: ' ', 2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j', 12: 'k', 
            13: 'l', 14: 'm', 15: 'n', 16: 'o', 17: 'p', 18: 'q', 19: 'r', 20: 's', 21: 't', 22: 'u', 23: 'v', 
            24: 'w', 25: 'x', 26: 'y', 27: 'z'}

class ASRModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.3, fil_dim=32):
        super(ASRModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, fil_dim, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(fil_dim),
            torch.nn.GELU(),            
            torch.nn.Conv2d(fil_dim, fil_dim*2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(fil_dim*2),
            torch.nn.GELU()
        )
        self.conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64*input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),  
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
        )
        self.bi_gru = torch.nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=True
        )
        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim*2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x, lengths):
        x = x.unsqueeze(1) 
        x = self.conv(x)  
        x = x.permute(0, 3, 2, 1)  
        x = x.reshape(x.size(0), x.size(1), -1)  
        x = x.permute(0, 2, 1)  
        x = self.conv1d(x)   
        x = x.permute(0, 2, 1)   
        lengths = torch.tensor(lengths).cpu().int()
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bi_gru(packed_x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(output)

def load_arpa_lm(arpa_file):
    lm, backoff = {}, {}
    with open(arpa_file, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()            
            if line.startswith("\\") or not line:
                continue           
            parts = line.split("\t")
            if len(parts) >= 2:
                prob = float(parts[0])
                ngram = parts[1]              
                if len(parts) == 3:  
                    backoff[ngram] = float(parts[2])                
                lm[ngram] = prob  
    return lm, backoff

def get_lm_score(seq, lm, backoff):
    if seq in lm:
        return lm[seq]
    if len(seq) > 1:
        backoff_word = seq[:-1]
        if backoff_word in backoff:
            return backoff[backoff_word] + get_lm_score(seq[1:], lm, backoff)
    return -10    

def get_best_lm_replacement(word, context, lm, backoff):
    best_score = float('-inf')
    best_word = word  
    closest_match = difflib.get_close_matches(word, lm.keys(), n=3, cutoff=0.8)
#     if closest_match:
#         best_word = closest_match[0]
#         best_score = lm[best_word]
#     else:
#         best_word = word
    for candidate in closest_match:
        candidate_ngram = context + " " + candidate
        score = get_lm_score(candidate_ngram, lm, backoff)
        if score > best_score:
            best_word = candidate
            best_score = score 
    return best_word

def post_process_with_lm(decoded_seq, lm, backoff):
    words = decoded_seq.strip().split()
    final_transcription = []
    for i, word in enumerate(words):
        word = word.strip()
        context = " ".join(final_transcription[max(0, i-1):i])            
        if word in lm: 
            final_transcription.append(word)  
        else:
            best_replacement = get_best_lm_replacement(word, context, lm, backoff)
            final_transcription.append(best_replacement)
    return " ".join(final_transcription)

def beam_search_decoder(probs, char_map, k, lm, backoff):
    T, V = probs.shape
    beam = [("", 0.0)]
    for t in range(T):
        new_beam = {}        
        for seq, log_probs in beam:
            for v in range(V):
                if v == 0: 
                    new_seq = seq  
                elif len(seq) > 0 and seq[-1] == char_map[v]:  
                    new_seq = seq
                else:
                    new_seq = seq + char_map[v]                
                new_probs = log_probs + probs[t, v]  
                if new_seq in new_beam:
                    new_beam[new_seq] = max(new_beam[new_seq], new_probs)
                else:
                    new_beam[new_seq] = new_probs
        beam = heapq.nlargest(k, new_beam.items(), key=lambda x: x[1])

    best_seq = beam[0][0]
    final_transcription = post_process_with_lm(best_seq, lm, backoff)    
    return final_transcription, best_seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim, hidden_dim, output_dim = 128, 512, 28
K = 5
model = ASRModel(input_dim, hidden_dim, output_dim).to(device)

############################################################################ ../models/augmented/conv1d/
model.load_state_dict(torch.load("asr_model--3.pth", map_location=device)) # here you can change model file
model.eval()
lm, backoff = load_arpa_lm("3gram.arpa")                                               # Load the Language Model file
AUDIO_FILE_PATH = "test audio/LJ001-0075.wav"                                                                             # Upload the audio file
############################################################################### ../kenlm/build/

def transcribe(audio_path):
    model.to(device)
    model.eval()
    audio, sr = librosa.load(audio_path, sr=16000)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    audio_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0).to(device)    
    with torch.no_grad():
        output = model(audio_tensor, [audio_tensor.shape[2]])
    output = torch.nn.functional.log_softmax(output, dim=-1).cpu().squeeze().numpy()    
    transcription, asr_transcript = beam_search_decoder(output, char_map, k=K, lm=lm, backoff=backoff)

    return transcription, asr_transcript

if __name__ == "__main__":
    transcription, asr_transcript = transcribe(AUDIO_FILE_PATH)
    
    print("\n=== ASR Transcript (Before LM) ===\n")
    print(asr_transcript)

    print("\n=== Final Transcript (After LM) ===\n")
    print(transcription)
