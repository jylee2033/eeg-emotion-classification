import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import Config
from data.feature_engineering import create_asymmetry_features

def load_and_engineer_data():
    print(f"Loading data from {Config.CSV_FILE_PATH}...")
    data = pd.read_csv(Config.CSV_FILE_PATH).dropna()

    engineered = create_asymmetry_features(data, Config.SYMMETRIC_PAIRS)

    labels = data["Emo_Label_Cowen(27)"].values - 1
    features = engineered.values
    feature_names = engineered.columns.tolist()

    print(f"After Feature Engineering, total features: {features.shape[1]}")

    return data, features, labels, feature_names


def scale_and_split(features, labels):
    print("Final Scaling + Train/Val Split...")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled,
        labels,
        test_size=0.1,
        random_state=Config.RANDOM_STATE,
        stratify=labels
    )

    return X_train, X_val, y_train, y_val
