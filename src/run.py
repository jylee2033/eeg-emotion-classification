import os
import numpy as np
import pandas as pd
import torch
import gradio as gr
from typing import List, Dict, Any
from math import sqrt, log10

from sklearn.preprocessing import StandardScaler

from config import Config
from model.cnn_eeg import EEG_CNN_Model
from data.feature_engineering import create_asymmetry_features

# Constants sourced from Config instead of deprecated emotional_classifier module
NUM_EMOTIONS = Config.NUM_EMOTIONS
NUM_CHANNELS = Config.NUM_CHANNELS
SYMMETRIC_PAIRS = Config.SYMMETRIC_PAIRS

CSV_FILE_PATH = Config.CSV_FILE_PATH
MODEL_PATH = Config.MODEL_SAVE_PATH

EMOTIONS = [
    "admiration", "adoration", "aesthetic appreciation", "amusement", "anger",
    "anxiety", "awe", "awkwardness", "boredom", "calmness", "confusion",
    "craving", "disgust", "empathic pain", "entrancement", "excitement",
    "fear", "horror", "interest", "joy", "nostalgia", "relief", "romance",
    "sadness", "satisfaction", "sexual desire", "surprise",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

all_features_tensor = None   # (N, F_per_channel, C)
cowen_labels_zero_based = None  # (N,) 0~26
cowen_labels_raw = None         # (N,) 1~27
participant_ids = None          # (N,)
num_samples = None
num_features_per_channel = None
model = None
training_scaler = None
training_engineered_columns = None  # list of engineered column names after asymmetry feature creation
training_base_feature_names = None  # original feature column names before reordering


def prepare_features():
    global all_features_tensor, \
       cowen_labels_zero_based, \
       cowen_labels_raw, \
       participant_ids, \
       num_samples, \
       num_features_per_channel, \
       training_scaler

    print(f"Loading CSV from {CSV_FILE_PATH} ...")
    data = pd.read_csv(CSV_FILE_PATH).dropna()

    cowen_labels_raw = data["Emo_Label_Cowen(27)"].values.astype(int)  # 1~27
    participant_ids = data["ParticipantID"].values.astype(int)

    cowen_labels_zero_based = cowen_labels_raw - 1

    data_engineered = create_asymmetry_features(data, SYMMETRIC_PAIRS)
    features = data_engineered.values  # (N, F_total)

    training_scaler = StandardScaler()
    features_scaled = training_scaler.fit_transform(features)

    num_samples_local, num_total_features = features_scaled.shape

    if num_total_features % NUM_CHANNELS != 0:
        raise ValueError(
            f"ValueError: Invalid feature layout — total features ({num_total_features}) are not divisible by the number of channels ({NUM_CHANNELS}). The dataset structure may be corrupted or mismatched."
        )

    n_feat_per_ch = num_total_features // NUM_CHANNELS

    # (N, F_total) -> (N, C, F_per_channel) -> (N, F_per_channel, C)
    features_grouped = features_scaled.reshape(
        num_samples_local, NUM_CHANNELS, n_feat_per_ch
    )
    features_transposed = np.transpose(features_grouped, (0, 2, 1))

    all_features_tensor = torch.tensor(
        features_transposed, dtype=torch.float32
    )  # (N, F_per_channel, C)

    num_samples = num_samples_local
    num_features_per_channel = n_feat_per_ch

    print(f"Prepared features: {all_features_tensor.shape}, samples: {num_samples}")


def _ar_coeffs(x: np.ndarray, order: int = 4) -> List[float]:
    n = len(x)
    if n <= order:
        return [0.0] * order
    X = np.column_stack([x[i: n - order + i] for i in range(order)])
    y = x[order:]
    # solve X a = y in least squares sense
    try:
        a, *_ = np.linalg.lstsq(X, y, rcond=None)
        return a.tolist()
    except Exception:
        return [0.0] * order


def _hjorth(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    if len(x) < 2:
        return {"ha": 0.0, "hm": 0.0, "hc": 0.0}
    var0 = float(np.var(x))
    var1 = float(np.var(dx))
    ha = var0
    hm = sqrt(var1 / var0) if var0 > 0 else 0.0
    ddx = np.diff(dx)
    var2 = float(np.var(ddx)) if len(ddx) > 0 else 0.0
    hm_der = sqrt(var2 / var1) if var1 > 0 else 0.0
    hc = hm_der / hm if hm > 0 else 0.0
    return {"ha": ha, "hm": hm, "hc": hc}


def _teager_energy(x: np.ndarray) -> Dict[str, float]:
    if len(x) < 3:
        return {"te": 0.0, "mte": 0.0}
    x = np.asarray(x, dtype=float)
    t = x[1:-1] * x[1:-1] - x[:-2] * x[2:]
    te = float(np.sum(t))
    mte = float(np.mean(t))
    return {"te": te, "mte": mte}


def _kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    m = np.mean(x)
    s2 = np.mean((x - m) ** 2)
    if s2 == 0:
        return 0.0
    return float(np.mean((x - m) ** 4) / (s2 ** 2))


def _skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _bandpowers(x: np.ndarray, fs: float) -> Dict[str, float]:
    # Welch PSD
    if len(x) < 8:
        return {"bpd": 0.0, "bpt": 0.0, "bpa": 0.0, "bpb": 0.0, "bpg": 0.0, "rba": 0.0}
    from numpy.fft import rfft, rfftfreq
    n = len(x)
    x = x - np.mean(x)
    window = np.hanning(n)
    X = rfft(x * window)
    freqs = rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(X) ** 2) / np.sum(window ** 2)

    def band_power(f_lo, f_hi):
        idx = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.trapz(psd[idx], freqs[idx])) if np.any(idx) else 0.0

    bpd = band_power(0.5, 4.0)
    bpt = band_power(4.0, 8.0)
    bpa = band_power(8.0, 13.0)
    bpb = band_power(13.0, 30.0)
    bpg = band_power(30.0, 45.0)
    total = bpd + bpt + bpa + bpb + bpg
    rba = (bpa / total) if total > 0 else 0.0
    return {"bpd": bpd, "bpt": bpt, "bpa": bpa, "bpb": bpb, "bpg": bpg, "rba": rba}


def _spectral_entropy(x: np.ndarray, fs: float) -> float:
    if len(x) < 8:
        return 0.0
    from numpy.fft import rfft, rfftfreq
    n = len(x)
    x = x - np.mean(x)
    window = np.hanning(n)
    X = rfft(x * window)
    psd = (np.abs(X) ** 2)
    psd /= psd.sum() if psd.sum() > 0 else 1.0
    # Shannon entropy
    eps = 1e-12
    return float(-np.sum(psd * np.log(psd + eps)))


def _infer_sampling_rate(n_samples: int) -> float:
    # heuristic based on dataset: many trials ~45-60s at 128 or 256 Hz
    # if very long, assume 256 Hz else 128 Hz
    return 256.0 if n_samples >= 9000 else 128.0


def _compute_channel_feature_dict(x: np.ndarray, fs: float) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    x = np.asarray(x, dtype=float)
    feats["min"] = float(np.min(x)) if x.size else 0.0
    feats["max"] = float(np.max(x)) if x.size else 0.0
    a1, a2, a3, a4 = _ar_coeffs(x, order=4)
    feats["ar1"], feats["ar2"], feats["ar3"], feats["ar4"] = a1, a2, a3, a4
    feats["md"] = float(np.median(x)) if x.size else 0.0
    feats["var"] = float(np.var(x)) if x.size else 0.0
    feats["sd"] = float(np.std(x)) if x.size else 0.0
    feats["am"] = float(np.mean(np.abs(x))) if x.size else 0.0
    feats["me"] = float(np.mean(x)) if x.size else 0.0
    feats["lrssv"] = log10(sqrt(np.sum(x * x)) + 1e-12) if x.size else 0.0

    # Teager Energy and spectral metrics
    feats.update(_teager_energy(x))  # adds 'te', 'mte'
    feats["le"] = float(np.sum(np.log(x * x + 1e-12))) if x.size else 0.0
    feats["sh"] = _spectral_entropy(x, fs)

    # Hjorth and bandpowers
    feats.update(_hjorth(x))  # adds 'ha','hm','hc'
    feats.update(_bandpowers(x, fs))  # adds bpd,bpt,bpa,bpb,bpg,rba

    # Simple difference-based measures as placeholders
    if x.size > 1:
        d1 = np.abs(np.diff(x))
        feats["1d"] = float(np.mean(d1))
        feats["n1d"] = float(np.sum(d1 > (np.std(x) if np.std(x) > 0 else 1.0)))
        d2 = np.abs(np.diff(x, n=2)) if x.size > 2 else np.array([0.0])
        feats["2d"] = float(np.mean(d2))
        feats["n2d"] = float(np.sum(d2 > (np.std(d1) if np.std(d1) > 0 else 1.0)))
    else:
        feats["1d"] = 0.0
        feats["n1d"] = 0.0
        feats["2d"] = 0.0
        feats["n2d"] = 0.0

    # higher-order moments
    feats["kurt"] = _kurtosis(x)
    feats["skew"] = _skewness(x)

    return feats


def extract_features_from_txt_files(paths: List[str]) -> pd.DataFrame:
    """Create a DataFrame with one row per raw EEG txt, columns matching training base schema.
    Unknown feature names will be filled with 0. Columns are ordered as in training CSV.
    """
    global training_base_feature_names
    rows: List[Dict[str, Any]] = []
    for p in paths:
        arr = np.loadtxt(p, ndmin=2)
        # Expect shape (T, 14)
        if arr.ndim != 2 or arr.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected a 2D array with {NUM_CHANNELS} channels per row. Got shape {arr.shape} for {p}")
        T = arr.shape[0]
        fs = _infer_sampling_rate(T)
        row: Dict[str, Any] = {}
        for ch in range(1, NUM_CHANNELS + 1):
            x = arr[:, ch - 1]
            ch_feats = _compute_channel_feature_dict(x, fs)
            for base in training_base_feature_names:
                key = f"{base}_{ch}"
                # supply computed feature if available, else 0.0
                row[key] = float(ch_feats.get(base, 0.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure all expected columns exist
    expected_cols = [f"{base}_{ch}" for base in training_base_feature_names for ch in range(1, NUM_CHANNELS + 1)]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0.0
    # Reorder
    df = df[expected_cols]
    return df


def load_model():
    global model, num_features_per_channel

    if all_features_tensor is None:
        prepare_features()

    print(f"Loading model from {MODEL_PATH} ...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"FileNotFoundError: The model file could not be found at: {MODEL_PATH}")

    m = EEG_CNN_Model(
        num_features=num_features_per_channel,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_EMOTIONS,
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()

    model = m
    print("Model loaded and ready.")


def predict(sample_index: int):
    if model is None:
        load_model()

    idx = int(sample_index)
    if idx < 0 or idx >= num_samples:
        return {"error": f"The index must be between 0 and {num_samples-1} (inclusive)."}, ""

    x = all_features_tensor[idx].unsqueeze(0).to(device)  # (1, F, C)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))            # 0~26
    true_idx = int(cowen_labels_zero_based[idx])  # 0~26

    prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(NUM_EMOTIONS)}

    pid = int(participant_ids[idx])
    cowen_raw = int(cowen_labels_raw[idx])      # 1~27

    info_text = (
        f"Sample #{idx}\n"
        f"- ParticipantID: {pid}\n"
        f"- Cowen Label: {cowen_raw} ({EMOTIONS[true_idx]})\n"
        f"- Predicted (Model): {EMOTIONS[pred_idx]}"
    )

    return prob_dict, info_text


def predict_uploaded_file(file):
    """
    Predict emotions from an uploaded file containing EEG data.
    Supports:
    - CSV files with pre-extracted features (same format as training data)
    - TXT files with raw EEG data (tab-separated, 14 channels)
    - Folders containing multiple TXT files
    """
    if model is None:
        load_model()

    try:
        # Gradio may pass a single file, list of files, or a tmp path
        if isinstance(file, list):
            files = file
        else:
            files = [file]

        # Flatten to file paths; handle gradio objects
        file_paths: List[str] = []
        for f in files:
            p = f.name if hasattr(f, 'name') else str(f)
            if os.path.isdir(p):
                # include all .txt files in directory
                for name in os.listdir(p):
                    if name.lower().endswith('.txt'):
                        file_paths.append(os.path.join(p, name))
            else:
                file_paths.append(p)

        # Branch based on extension of the first file
        first_ext = os.path.splitext(file_paths[0])[1].lower() if file_paths else ''
        
        # Handle CSV files
        if first_ext == '.csv':
            # Read the uploaded CSV
            uploaded_frames = [pd.read_csv(p) for p in file_paths]
            uploaded_data = pd.concat(uploaded_frames, ignore_index=True).dropna()
            
            # Apply the same feature engineering (asymmetry features)
            data_engineered = create_asymmetry_features(uploaded_data, SYMMETRIC_PAIRS)
            features = data_engineered.values  # (N, F_total)
            
            # Scale features using StandardScaler
            if training_scaler is None:
                return {"error": 1.0}, "Training scaler not initialized. Please ensure training features are prepared."
            features_scaled = training_scaler.transform(features)
            
            num_samples_uploaded, num_total_features = features_scaled.shape
            
            # Validate feature dimensions
            if num_total_features % NUM_CHANNELS != 0:
                return {"error": 1.0}, f"Error: Feature count ({num_total_features}) is not divisible by channels ({NUM_CHANNELS})."
            
            n_feat_per_ch = num_total_features // NUM_CHANNELS
            
            # Reshape: (N, F_total) -> (N, C, F_per_channel) -> (N, F_per_channel, C)
            features_grouped = features_scaled.reshape(
                num_samples_uploaded, NUM_CHANNELS, n_feat_per_ch
            )
            features_transposed = np.transpose(features_grouped, (0, 2, 1))
            features_tensor = torch.tensor(features_transposed, dtype=torch.float32).to(device)
            
            # Run inference on all samples
            with torch.no_grad():
                logits = model(features_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Get predictions for all samples
            pred_indices = np.argmax(probs, axis=1)
            
            # Format results
            results_text = f"Processed {num_samples_uploaded} sample(s) from uploaded CSV.\n\n"
            
            # Show detailed results for each sample
            for i in range(min(num_samples_uploaded, 10)):  # Show up to 10 samples
                pred_idx = int(pred_indices[i])
                confidence = float(probs[i][pred_idx])
                results_text += f"Sample {i+1}: {EMOTIONS[pred_idx]} (confidence: {confidence:.2%})\n"
            
            if num_samples_uploaded > 10:
                results_text += f"\n... and {num_samples_uploaded - 10} more samples"
            
            # Return the probabilities of the first sample for the Label component
            first_sample_probs = {EMOTIONS[i]: float(probs[0][i]) for i in range(NUM_EMOTIONS)}
            
            return first_sample_probs, results_text
        
        # Handle TXT files (raw EEG data)
        elif first_ext == '.txt' or first_ext == '':
            # Extract features from raw EEG .txt files
            if training_base_feature_names is None or training_engineered_columns is None:
                return {"error": 1.0}, "Training feature schema not initialized. Please reload and try again."

            uploaded_base = extract_features_from_txt_files(file_paths)
            # Engineer asymmetry features exactly as in training
            engineered = create_asymmetry_features(uploaded_base, SYMMETRIC_PAIRS)

            # Align columns to training engineered order; fill missing with 0
            for c in training_engineered_columns:
                if c not in engineered.columns:
                    engineered[c] = 0.0
            engineered = engineered[training_engineered_columns]

            features = engineered.values
            if training_scaler is None:
                return {"error": 1.0}, "Training scaler not initialized. Please ensure training features are prepared."
            features_scaled = training_scaler.transform(features)

            num_samples_uploaded, num_total_features = features_scaled.shape
            # Validate dimensions match model expectation per channel
            if num_total_features % NUM_CHANNELS != 0:
                return {"error": 1.0}, f"Error: Feature count ({num_total_features}) is not divisible by channels ({NUM_CHANNELS})."
            n_feat_per_ch = num_total_features // NUM_CHANNELS
            if n_feat_per_ch != num_features_per_channel:
                return {"error": 1.0}, (
                    f"Feature dimensionality mismatch: uploaded per-channel features = {n_feat_per_ch}, "
                    f"but model expects {num_features_per_channel}."
                )

            # Reshape and predict
            features_grouped = features_scaled.reshape(num_samples_uploaded, NUM_CHANNELS, n_feat_per_ch)
            features_transposed = np.transpose(features_grouped, (0, 2, 1))
            features_tensor = torch.tensor(features_transposed, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(features_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            pred_indices = np.argmax(probs, axis=1)
            results_text = f"Processed {num_samples_uploaded} raw EEG file(s).\n\n"
            for i in range(min(num_samples_uploaded, 10)):
                pred_idx = int(pred_indices[i])
                confidence = float(probs[i][pred_idx])
                base_name = os.path.basename(file_paths[i]) if i < len(file_paths) else f"Sample {i+1}"
                results_text += f"{base_name}: {EMOTIONS[pred_idx]} (confidence: {confidence:.2%})\n"
            if num_samples_uploaded > 10:
                results_text += f"\n... and {num_samples_uploaded - 10} more samples"

            first_sample_probs = {EMOTIONS[i]: float(probs[0][i]) for i in range(NUM_EMOTIONS)}
            return first_sample_probs, results_text
        
        # Handle other file types
        else:
            return {"error": 1.0}, f"Unsupported file type(s). Please upload CSV feature files or raw EEG .txt files."
        
    except Exception as e:
        return {"error": 1.0}, f"Error processing uploaded file: {str(e)}"


def main():
    try:
        prepare_features()
        load_model()
    except Exception as e:
        print(f"Warning during initialization: {e}")
        print("Gradio will still launch. You can test with an index later to check for errors.")


    index_input = gr.Slider(
        minimum=0,
        maximum=(num_samples - 1) if num_samples is not None else 100,
        step=1,
        value=0,
        label="Sample Index (0-based)",
    )

    output_label = gr.Label(
        num_top_classes=5,
        label="Predicted Emotion Probabilities",
    )

    output_text = gr.Textbox(
        label="Sample Info (Participant / Labels / Prediction)",
        lines=6,
    )

    # Create tabbed interface with Blocks for better layout control
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("User Upload Prediction"):
                gr.Markdown("# EEG Emotion Classifier")
                gr.Markdown(
                    "Upload a file containing EEG data.\n\n"
                    "It must be in **CSV Format** with pre-extracted features matching the format of the training data (refer to sample dataset):\n"
                    "- 14 channels with ~35 features each per sample\n"
                    "- Asymmetry features computed automatically\n\n"
                    "The model will present a detailed confidence distribution for the first sample, along with confidence scores for the rest of the samples.\n\n"
                    "Below is a sample dataset you can download and test with."
                )
                
                # Sample download file placed right after description
                sample_file = gr.File(
                    label="Sample Dataset",
                    value="src/data/sample_eeg_dataset.csv",
                    interactive=False,
                )
                
                # Upload section
                file_input = gr.File(
                    label="Upload EEG Data File",
                    file_types=[".csv", ".txt"],
                )
                
                # Outputs
                output_label_2 = gr.Label(
                    num_top_classes=5,
                    label="Predicted Emotion Confidence Distribution (First Sample)",
                )
                output_text_2 = gr.Textbox(
                    label="Prediction Results (Following Samples)",
                    lines=10,
                )
                
                file_input.change(
                    fn=predict_uploaded_file,
                    inputs=file_input,
                    outputs=[output_label_2, output_text_2]
                )
            
            with gr.Tab("Training Data Viewer"):
                gr.Markdown("# Browse Training Data Samples")
                gr.Markdown(
                    "Select the N-th sample from the `eeg_features_extracted.csv` file to see how the trained CNN model predicts its emotion.\n\n"
                    "The interface also displays the Participant ID and Cowen (27-class), along with the model's prediction."
                )
                
                index_input_tab = gr.Slider(
                    minimum=0,
                    maximum=(num_samples - 1) if num_samples is not None else 100,
                    step=1,
                    value=0,
                    label="Sample Index (0-based)",
                )
                
                output_label_tab = gr.Label(
                    num_top_classes=5,
                    label="Predicted Emotion Probabilities",
                )
                output_text_tab = gr.Textbox(
                    label="Sample Info (Participant / Labels / Prediction)",
                    lines=6,
                )
                
                index_input_tab.change(
                    fn=predict,
                    inputs=index_input_tab,
                    outputs=[output_label_tab, output_text_tab]
                )

    demo.launch()


if __name__ == "__main__":
    main()
