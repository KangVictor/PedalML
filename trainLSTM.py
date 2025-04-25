import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchaudio
from models.FxNet import FxNet
from models.LSTM import LSTMModel
from utils.data_pre import match_list_lengths, get_wav_file_list
from utils.audio_pre import pre_emphasis
from torch.utils.data import Dataset, DataLoader
from dataset import GuitarEffectsDataset
from os import listdir
from os.path import isfile, join

# Initialize model and optimizer
# 1. Load FXNet model as an encoding model
model_path = "models/fx_classifier_models/fxnet_poly_cont_best.pth"
n_classes = 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fxnet_model = FxNet(n_classes=n_classes)
fxnet_model.load_state_dict(torch.load(model_path, map_location=device))
fxnet_model.to(device)
# print(fxnet_model.eval())

# 2. LSTM model to Train
model = LSTMModel(input_size=1, hidden_size=128, num_layers = 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3. mel_spectrogram to apply FXNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a Mel-spectrogram transform (differentiable)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100,       # sample rate of audio
    n_fft=2048,             # FFT size
    hop_length=256,         # hop size for STFT
    n_mels=128              # number of Mel frequency bands
).to(device)


# 2. Prepare dataset for training
clean_file_path = "data/input"
clean_files = get_wav_file_list(clean_file_path)
effect_file_path = "data/target/desert_eagle"
effect_files = get_wav_file_list(effect_file_path)

print("# Clean files: ", len(clean_files))
print(clean_files[:5])
print("# Effect files", len(effect_files))
print(effect_files[:5])

clean_files, effect_files = match_list_lengths(clean_files, effect_files)
dataset = GuitarEffectsDataset(clean_files, effect_files, sample_rate=44100, segment_length=3.0)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# 3. Start training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
num_epochs = 30

# Lists to track loss
epoch_losses = []
spectral_losses = []
embedding_losses = []

for epoch in range(num_epochs):
    total_loss, total_spectral, total_embedding = 0.0, 0.0, 0.0

    for clean_wave, target_wave in loader:
        clean_wave = clean_wave.to(device)
        target_wave = target_wave.to(device)

        # Forward pass
        # clean_wave = clean_wave.transpose(1, 2) # Due to LSTM should look like (B, T, 1)
        output_wave, _ = model(clean_wave.transpose(1, 2))
        output_wave = output_wave.transpose(1, 2)  # LSTM output shape: (B, T, 1) → (B, 1, T)

        # Spectral loss (pre-emphasized MSE)
        clean_pre = pre_emphasis(clean_wave)
        output_pre = pre_emphasis(output_wave)
        
        spectral_loss = F.mse_loss(output_pre, clean_pre)

        # Embedding loss (MAE in fxnet_model feature space)
        mel_target = mel_transform(target_wave)
        mel_output = mel_transform(output_wave)

        if mel_target.dim() == 3:
            mel_target = mel_target.unsqueeze(1)
            mel_output = mel_output.unsqueeze(1)



        # # Interpolate for 3 times
        # mel_target = mel_target.repeat(1, 1, 1, 3)
        # mel_output = mel_output.repeat(1, 1, 1, 3)




        with torch.no_grad():  # fxnet_model is frozen, no gradients needed
            emb_target = fxnet_model(mel_target)

        emb_output = fxnet_model(mel_output)
        embedding_loss = F.l1_loss(emb_output, emb_target)

        # Total loss
        loss = spectral_loss + embedding_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_spectral += spectral_loss.item()
        total_embedding += embedding_loss.item()

    # Average losses
    avg_loss = total_loss / len(loader)
    avg_spectral = total_spectral / len(loader)
    avg_embedding = total_embedding / len(loader)

    epoch_losses.append(avg_loss)
    spectral_losses.append(avg_spectral)
    embedding_losses.append(avg_embedding)

    clear_output(wait=True)
    print(f"Epoch {epoch+1}/{num_epochs} - Total Loss: {avg_loss:.4f}, "
          f"Spectral: {avg_spectral:.4f}, Embedding: {avg_embedding:.4f}")

    # Intermediate visualization every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_losses, label='Total Loss')
        plt.plot(spectral_losses, label='Spectral Loss', linestyle='--')
        plt.plot(embedding_losses, label='Embedding Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid()
        plt.show()

# Final results
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label='Total Loss')
plt.plot(spectral_losses, label='Spectral Loss', linestyle='--')
plt.plot(embedding_losses, label='Embedding Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final Training Losses')
plt.legend()
plt.grid()
plt.show()
