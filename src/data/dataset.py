
import torch
from torch.utils.data import Dataset
from config import Config

class EEGDataset(Dataset):
  def __init__(self, features, labels, num_channels=Config.NUM_CHANNELS):
    num_samples = features.shape[0]
    num_total_features = features.shape[1]
    
    self.num_features_per_channel = num_total_features // num_channels
    if num_total_features % num_channels != 0:
        raise ValueError(f"Total features ({num_total_features}) not divisible by channels ({num_channels}). Check feature ordering.")

    features_grouped_by_channel = features.reshape(
        num_samples, num_channels, self.num_features_per_channel
    )
    features_transposed = features_grouped_by_channel.transpose(0, 2, 1)

    self.features = torch.tensor(features_transposed, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.long)
    print(f"Dataset ready")

  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]