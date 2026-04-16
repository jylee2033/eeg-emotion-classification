# EEECDL (Electroencephalo-emotional Classifer via Deep Learning)
## Contributors
- Richard Gao @MrFlyingPizza
- Jooyoung (Julia) Lee @jylee2033
- Calvin Weng @yamikazoo
- Aarham Haider @AarhamH
- Abrar Rahman @abr-rhmn

## Important Links
| [Project report](https://www.overleaf.com/9314115499gkgmhkrgjdzm#801001) |
|-----------|

## Video/demo/GIF
https://youtu.be/qVf95Jlxs44 

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

Training Loop
![train_record](https://github.com/user-attachments/assets/61403f00-36dd-461c-959e-88b6de3bbe88)

Sample Charts

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/29d4ce4d-3fe1-4455-ba51-9240f4bc2637" />
<img width="300" height="800" alt="image" src="https://github.com/user-attachments/assets/3838322e-397d-4b85-bbfd-6f2ea2f49c93" />
<img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/3520ceb0-cc9b-4eca-bffb-929466721b21" />




### What to find where

```bash
repository
├── src                          ## source code of the package itself
    ├── data                     ## code for data preprocessing, feature engineering, and the EEGDataSet class
    ├── model                    ## code for 1-D CNN class and training
    ├── utils                    ## utility
    config.py                    ## contains paths and hyperparameters used by learning model
    main.py                      ## main driver
    run.py                       ## gradio deliverable
├── README.md                    ## You are here
├── requirements.txt             ## If you use conda
```

<a name="installation"></a>

## 2. Installation and Training

Install the project.

1. Create the virtual environment
   ```shell
   python -m venv .venv
   ```
2. Activate the virtual environment (this may take a while on CSIL)

    **macOS / Linux**
   ```shell
   source .venv/bin/activate
   ```
    **Windows**
   ```shell
    .venv\Scripts\activate
   ```
2. Install dependencies using the requirements file (there are many to install so this may take additional time). 
   ```shell
   pip install -r requirements.txt <---- this will be found in this repo, so cd into the repo first
   ```
3. If you update the dependencies, remember to freeze it and commit the changes to the requirements.txt everytime.
   ```shell
   pip freeze > requirements.txt
   ```



<a name="repro"></a>
## 3. Reproduction
Clone the repository
```bash
git clone https://github.com/huytungst/EEGEmotions-27.git
cd EEGEmotions-27
```
Windows Powershell
````powershell
Move-Item -Path "path\to\EEGEmotions-27\training\eeg_features_extracted.csv" -Destination "path\to\2025_3_project_06\src\"
````

Linux/macOS
````bash
mv /path/to/EEGEmotions-27/training/eeg_features_extracted.csv /path/to/2025_3_project_06/src/
````

If you wish the rename the file, or change the path overall, you have to change `CSV_FILE_PATH` parameter under `src/config.py`
````python
class Config:
...
CSV_FILE_PATH = "eeg_features_extracted.csv" <---- CHANGE YOUR .csv PATH HERE
 ...
````

To train the model, simply cd into the repo and run
````bash
python src/main.py 
````
After training has completed, a set of charts will be created under `plots/` for accuracy and loss for training and validation loops, as well as a confusion matrix.

Data can be found at: [https://github.com/huytungst/EEGEmotions-27](https://github.com/huytungst/EEGEmotions-27)

Output will be saved in: `best_cnn_model.pth` (our model with the best model weights)

<a name="gradio"></a>
## 4. Using the Gradio Web Interface

After training the model, you can use the interactive Gradio web application to make predictions on new EEG data.

### Running the Application

1. Ensure your virtual environment is activated (see step 2 in Installation)

2. From the project root directory, run:
   ```bash
   python src/run.py
   ```

3. The application will start and display a local URL (typically `http://127.0.0.1:7860`)

4. Open the URL in your web browser

### Using the Interface

The application has two tabs:

#### **User Upload Prediction** (Default Tab)
Upload your own EEG data files to get emotion predictions.

- **Download Sample Dataset**: Click the download button to get a sample CSV file (`sample_eeg_dataset.csv`) that demonstrates the correct format
- **Upload EEG Data File**: Upload a CSV file with pre-extracted EEG features
  - Must contain 14 channels with ~35 features each per sample
  - Asymmetry features are computed automatically
  - Supported formats: `.csv`
- **Results**: The model will display:
  - Predicted emotion probabilities for the first sample (top 5 emotions shown)
  - Confidence scores for all uploaded samples (up to 10 samples shown in detail)

#### **Training Data Viewer** (Second Tab)
Browse predictions on the training dataset to see how the model performs.

- Use the slider to select a sample index (0 to number of samples - 1)
- View:
  - Ground truth emotion label from the training data
  - Model's predicted emotion
  - Prediction confidence scores (top 5 emotions)
  - Participant ID and Cowen label information

### CSV Format Requirements

Your uploaded CSV file should match the training data format:
- **Columns**: Features for 14 EEG channels (e.g., `min_1`, `max_1`, `mean_1`, `ar1_1`, ..., `min_14`, `max_14`, etc.)
- **Rows**: Each row represents one EEG sample
- **No labels needed**: The model will predict the emotion labels

**Tip**: Download and examine the sample dataset from the web interface to see the exact format required.

