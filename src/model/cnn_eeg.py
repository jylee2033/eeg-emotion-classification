import torch.nn as nn
class EEG_CNN_Model(nn.Module):
  def __init__(self, num_features, num_channels, num_classes):
    super(EEG_CNN_Model, self).__init__()
    
    self.conv_block1 = nn.Sequential(
        nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(64)
    )
    
    self.conv_block2 = nn.Sequential(
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(128)
    )
    
    self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    self.flatten = nn.Flatten()
    
    self.fc = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, num_classes)
    )

  def forward(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.global_avg_pool(x)
    x = self.flatten(x)
    x = self.fc(x)
    return x