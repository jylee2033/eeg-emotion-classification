import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import Config
from data.preprocessing import load_and_engineer_data, scale_and_split
from data.feature_engineering import select_important_features
from data.dataset import EEGDataset
from model.cnn_eeg import EEG_CNN_Model
from model.trainer import train_model
from torch.utils.data import DataLoader
import torch

device = Config.DEVICE
print("Using device:", device)


if __name__ == "__main__":

    #first, load data
    data, features, labels, feature_names = load_and_engineer_data()

    # then select features via random forest
    selected_features = select_important_features(features, labels, feature_names)

    # scale data
    X_train, X_val, y_train, y_val = scale_and_split(features, labels)

    train_dataset = EEGDataset(X_train, y_train, num_channels=Config.NUM_CHANNELS)
    val_dataset   = EEGDataset(X_val, y_val, num_channels=Config.NUM_CHANNELS)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=Config.BATCH_SIZE, shuffle=False)

    num_features_per_channel = X_train.shape[1] // Config.NUM_CHANNELS

    model = EEG_CNN_Model(
        num_features=num_features_per_channel,
        num_channels=Config.NUM_CHANNELS,
        num_classes=Config.NUM_EMOTIONS
    ).to(device)

    train_model(model, train_loader, val_loader)
