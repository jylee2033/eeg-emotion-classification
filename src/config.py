import torch


class Config:
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5 
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    NUM_EMOTIONS = 27
    NUM_CHANNELS = 14
    CSV_FILE_PATH = "src/eeg_features_extracted.csv" 
    MODEL_SAVE_PATH = "best_cnn_model.pth"
    RANDOM_STATE = 42
    SYMMETRIC_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
    PLOT_DIR = "./plots"

    def get_device():
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    
    DEVICE = get_device()