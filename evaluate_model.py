
#####Now that the studying is done, the model now tests itself with a new batch of data

import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from train_model import ASVSpoofDataset, SpoofNet



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Uses computers GPU if possible
    
    #testing against new data
    eval_audio_dir = r"C:\Users\8reec\OneDrive\VoiceSpoofDetecter\data\raw\ASVspoof2019_LA\LA\LA\ASVspoof2019_LA_dev\flac"
    eval_protocol_path = r"C:\Users\8reec\OneDrive\VoiceSpoofDetecter\data\raw\ASVspoof2019_LA\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
    
    
    print("Loading evaluation dataset...")
    eval_dataset = ASVSpoofDataset(eval_protocol_path, eval_audio_dir, max_samples=64000) #4 second samples
    
    #this batch size can be larger since it is less deal on the GPU
    eval_loader = DataLoader(eval_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)
    
    lfcc_transform = T.LFCC(
        sample_rate=16000,
        n_filter=40, 
        speckwargs={"n_fft": 512, "hop_length": 160}
    )
    

    model = SpoofNet().to(device)
    
    # Load the saved weights
    print("Loading saved model weights...")
    model.load_state_dict(torch.load('spoofnet_weights.pth'))
    model.eval() #switches model from learning to testing mode
    
   
    print("Starting Evaluation...\n")
    
    correct_predictions = 0
    total_predictions = 0
    
    
    with torch.no_grad(): #turning this off stops memory tracking which executes the code faster
        for batch_idx, (waveforms, labels) in enumerate(eval_loader):

            #mostly the same code from first section
            features = lfcc_transform(waveforms)
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(features)
        
            predicted_classes = (outputs > 0.0).float()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Processed Batch {batch_idx}/{len(eval_loader)}")


    final_accuracy = (correct_predictions / total_predictions) * 100
    print("\n" + "="*40)
    print(f"Total Files Tested: {total_predictions}")
    print(f"Files Guessed Correctly: {correct_predictions}")
    print(f"Real-World Accuracy: {final_accuracy:.2f}%")
    print("="*40)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()