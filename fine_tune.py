
#######Add more training to model

import os
import torch
import torchaudio
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from train_model import SpoofNet


class MyVoiceDataset(Dataset):
    def __init__(self, real_dir, spoof_dir, max_samples=64000):
        self.max_samples = max_samples
        self.files = []
        self.labels = []
        
        # load real files with key of 1
        for f in os.listdir(real_dir):
            if f.endswith(('.flac', '.wav')):
                self.files.append(os.path.join(real_dir, f))
                self.labels.append(1.0)
                
        # load real files with key of 0
        for f in os.listdir(spoof_dir):
            if f.endswith(('.flac', '.wav')):
                self.files.append(os.path.join(spoof_dir, f))
                self.labels.append(0.0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Stereo to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample to 16,000 Hz
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            
        # Padding or snipping
        num_samples = waveform.shape[1]
        if num_samples < self.max_samples:
            waveform = F.pad(waveform, (0, self.max_samples - num_samples))
        elif num_samples > self.max_samples:
            waveform = waveform[:, :self.max_samples]
            
        return waveform.squeeze(0), torch.tensor(label, dtype=torch.float32)
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # directories of both folders
    real_dir = r"C:\Users\8reec\OneDrive\VoiceSpoofDetecter\my_real_voice"
    spoof_dir = r"C:\Users\8reec\OneDrive\VoiceSpoofDetecter\my_fake_voice"
    
    personal_dataset = MyVoiceDataset(real_dir, spoof_dir)
    train_loader = DataLoader(personal_dataset, batch_size=2, shuffle=True, num_workers=0) #doesn't need to be higher num workers since low amount of files

    # Load the model and previous weights
    model = SpoofNet().to(device)
    model.load_state_dict(torch.load('spoofnet_weights.pth', weights_only=True))
    
    # Uses lr as 0.0001 (10x smaller than before) so it doesn't "forget" the ASVspoof data
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    lfcc_transform = T.LFCC(
        sample_rate=16000, n_filter=40, 
        speckwargs={"n_fft": 512, "hop_length": 160}
    )

    # Train for only 3 epochs because of small file amount
    num_epochs = 3
    model.train()
    
    print(f"Fine-tuning on {len(personal_dataset)} personal audio files...")
    
    #previous training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for waveforms, labels in train_loader:
            
            features = lfcc_transform(waveforms).to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")

    # Save the newly adjusted brain as a different file so we don't overwrite the original
    torch.save(model.state_dict(), 'spoofnet_finetuned_weights.pth')
    print("Fine-tuning complete! Saved as 'spoofnet_finetuned_weights.pth'")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()