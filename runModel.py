import torch
import torchaudio
import torchaudio.transforms as T
from models.WaveNet import WaveNetModel 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load audio file
def load_audio(file_path, target_sr=44100):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        resampler = T.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    return waveform.to(device)

# Save audio file
def save_audio(file_path, waveform, sample_rate=44100):
    waveform_cpu = waveform.cpu().detach()
    torchaudio.save(file_path, waveform_cpu, sample_rate)

# Run inference on audio file
def process_audio(input_path, output_path):
    with torch.no_grad():
        input_wave = load_audio(input_path)  # shape: [1, L]
        input_wave = input_wave.unsqueeze(0)  # add batch dimension [1, 1, L]

        # Generate processed audio
        output_wave = model(input_wave)

        # Remove batch dimension
        output_wave = output_wave.squeeze(0)

        # Save output
        save_audio(output_path, output_wave)


model = WaveNetModel(residual_channels=32, skip_channels=32, dilation_layers=10)
model.load_state_dict(torch.load("models/wavenet_models/wavenet_weights.pth"))
model.eval()

# Example usage:
input_audio_file = "data/input.wav"
output_audio_file = "data/result.wav"

process_audio(input_audio_file, output_audio_file)

print(f"Audio processed and saved as '{output_audio_file}'.")
