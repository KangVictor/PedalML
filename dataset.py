import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class GuitarEffectsDataset(Dataset):
    def __init__(self, clean_files, effect_files, sample_rate=44100, segment_length=3.0):
        assert len(clean_files) == len(effect_files), "Mismatched number of files"
        self.clean_files = clean_files
        self.effect_files = effect_files
        self.target_sr = sample_rate
        self.segment_samples = int(sample_rate * segment_length)  # e.g., 3s segment

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load clean and effected audio
        clean_wave, sr_c = torchaudio.load(self.clean_files[idx])
        effect_wave, sr_e = torchaudio.load(self.effect_files[idx])
        # If stereo, take only one channel (assuming [channel, time] format from torchaudio)
        clean_wave = clean_wave.mean(dim=0, keepdim=True)  # shape [1, samples]
        effect_wave = effect_wave.mean(dim=0, keepdim=True)
        # Resample if needed to target sample rate
        if sr_c != self.target_sr:
            clean_wave = torchaudio.functional.resample(clean_wave, sr_c, self.target_sr)
        if sr_e != self.target_sr:
            effect_wave = torchaudio.functional.resample(effect_wave, sr_e, self.target_sr)
        # Pad or truncate to segment_samples
        # If too long, truncate; if too short, pad with zeros at end.
        if clean_wave.size(1) > self.segment_samples:
            clean_wave = clean_wave[:, :self.segment_samples]
            effect_wave = effect_wave[:, :self.segment_samples]
        else:
            pad_amount = self.segment_samples - clean_wave.size(1)
            if pad_amount > 0:
                clean_wave = F.pad(clean_wave, (0, pad_amount))
                effect_wave = F.pad(effect_wave, (0, pad_amount))
        # Return waveforms as tensors (shape [1, L]) and ensure dtype float32
        return clean_wave.float(), effect_wave.float()
