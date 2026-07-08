
#########tests against a new specific file of choice

import torch
import torchaudio
import torchaudio.transforms as T
import sys
from train_model import SpoofNet

def predict_audio(file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SpoofNet().to(device)
    model.load_state_dict(torch.load('spoofnet_finetuned_weights.pth', weights_only=True))
    model.eval() # Set to evaluation mode
    
    #setup the exact same transform from training
    lfcc_transform = T.LFCC(
        sample_rate=16000,
        n_filter=40, 
        speckwargs={"n_fft": 512, "hop_length": 160}
    )
    
    print(f"Analyzing: {file_path}...")
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1: #bug fix where it now formats the audio to have 1 channel (phone and comp. microphones record with 2 channels)
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != 16000: #bug fix 2 to where it auto formats to 16,000 Hz instead of higher from microphone recordings
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
        
    num_samples = waveform.shape[1]
    max_samples = 64000
    
    #strips down audio to 4 seconds or pads if shorter
    if num_samples < max_samples:
        pad_amount = max_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > max_samples:
        waveform = waveform[:, :max_samples]
        
    waveform = waveform.squeeze(0).unsqueeze(0) # Format to [1, 64000] to fix dimension mismatch error
    
    
    with torch.no_grad(): #fast outputs
        features = lfcc_transform(waveform).to(device)
        output = model(features)
        
        # Convert raw output to a percentage range (0.0 to 1.0)
        probability = torch.sigmoid(output).item() 
        
        print("-" * 40)
        if probability > 0.5:
            print(f"Result: REAL")
            print(f"Confidence: {probability * 100:.2f}%")
        else:
            print(f"Result: AI GENERATED")
            print(f"Confidence: {(1 - probability) * 100:.2f}%")
        print("-" * 40)

if __name__ == '__main__':
    #testing real and fake audio of me
    print("--- TESTING REAL VOICE ---")
    predict_audio(r"C:\Users\8reec\OneDrive\VoiceSpoofDetecter\real_me.flac")
    
    print("\n--- TESTING AI VOICE ---")
    predict_audio(r"C:\Users\8reec\OneDrive\VoiceSpoofDetecter\fake_me.flac")