
#####Training/studying phase of the model to learn from mistakes

import os
import torch
import torchaudio
import pandas as pd
import torch.nn as nn #neural network
import torch.optim as optim
import torch.nn.functional as F #contains math functions for processing
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


class ASVSpoofDataset(Dataset): #defines it as a Dataset class
    def __init__(self, protocol_path, audio_dir, max_samples=64000): #64,000 is 4 seconds
        #uses pandas to read spoof txt file and puts them into associated variables
        self.df = pd.read_csv(protocol_path, sep=" ", header=None, names=["speaker_id", "file_name", "system_id", 
                                                                          "null", "label"])
        self.audio_dir = audio_dir
        self.max_samples = max_samples 

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):

        #searches for each file and associates spoofed or bonafide with 1.0 and 0.0
        row = self.df.iloc[idx]
        file_name = row['file_name']
        label = 1.0 if row['label'] == 'bonafide' else 0.0

        #finds the audo flac file path
        full_path = os.path.join(self.audio_dir, f"{file_name}.flac")
        waveform, _ = torchaudio.load(full_path) # Shape: [1, num_samples]
        
        # formats the audio to make sure it is exactly 4 seconds, crops if over, if less it pads with silence
        num_samples = waveform.shape[1]  
        if num_samples < self.max_samples:
            pad_amount = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif num_samples > self.max_samples:
            waveform = waveform[:, :self.max_samples]
            
        # Returns the waveform and the answer key associated with it
        return waveform.squeeze(0), torch.tensor(label, dtype=torch.float32)


class SpoofNet(nn.Module):
    def __init__(self):
        super(SpoofNet, self).__init__()
        
        # Convolutional Block 1 that applies 16 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2 that applies additional 32 filters to find more complex features
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # formats the new data into 1D and connects 128 neurons to it
        self.fc1 = nn.Linear(32 * 10 * 100, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1) #formatting it into 4D (1 for color) so PyTorch can work correctly
        
        x = self.pool1(F.relu(self.conv1(x))) # Run the audio data through both convolutional/pooling blocks
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the 2D grid into a 1D vector
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x #sends final score back to training loop


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}") #This checks if NVIDIA CUDA is available and routes all the heavy lifting over to the GPU.
    
    audio_dir = r"C:\Users\8reec\OneDrive\VoiceSpoofDetecter\data\raw\ASVspoof2019_LA\LA\LA\ASVspoof2019_LA_train\flac" 
    protocol_path = r"C:\Users\8reec\OneDrive\VoiceSpoofDetecter\data\raw\ASVspoof2019_LA\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt" 
    
    
    train_dataset = ASVSpoofDataset(protocol_path, audio_dir, max_samples=64000)
    train_loader = DataLoader( #Loads files and shuffles them and groups them into groups of 256
        train_dataset, 
        batch_size=256, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=8
    )
    
    
    model = SpoofNet().to(device)
    criterion = nn.BCEWithLogitsLoss() #checks for loss with Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001) #uses Adam, with low learning rate. Tells the model to change weight to do better next time
    
    # Keep transform on the CPU to prevent Windows NVRTC dll errors
    lfcc_transform = T.LFCC( #LFCC turns soundwave into a spectogram like grid
        sample_rate=16000,
        n_filter=40, 
        speckwargs={"n_fft": 512, "hop_length": 160}
    )
    
    num_epochs = 5 #amount of times the data gets passed through the neural network
    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (waveforms, labels) in enumerate(train_loader):

            #grabs 256 files, runs the LFCC math, then sends matrices over to GPU
            features = lfcc_transform(waveforms)
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad() #Wipes the memory of the last batch's mistakes so they don't stack up
            
            #GPU pushes the 256 audio files through the CNN to get 256 guesses for each file
            labels = labels.unsqueeze(1) 
            outputs = model(features)
            
            #compares the guesses with the real answer
            loss = criterion(outputs, labels)
            loss.backward() #Runs backwards through the network to find which layers caused the wrong answer
            optimizer.step() #tweaks the model's memory to fix the mistakes
            running_loss += loss.item()
            
            #turns language into easy to read outputs
            predicted_classes = (outputs > 0.0).float()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Print update every 50 batches for progress
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct_predictions / total_predictions) * 100
        
        print(f"=== End of Epoch {epoch+1} ===")
        print(f"Average Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%\n")

    print("Training Complete!")

    # Save the trained weights to a file in the hard drive
    torch.save(model.state_dict(), 'spoofnet_weights.pth')
    print("Model saved.")


if __name__ == '__main__':
    # num_workers can cause scripts on Windows to crash, this stops that from happening
    torch.multiprocessing.freeze_support()
    main()
