# Deep Learning Experiments: CNN, RNN, and Transformer


https://drive.google.com/drive/folders/1mX_aMKFlJgFLaFf25atJjIK9aV4v4C5G?usp=sharing


A collection of deep-learning notebooks covering several supervised-learning and sequence-modeling tasks. The repository includes experiments with convolutional neural networks, recurrent neural networks, bidirectional LSTMs, and a T5 Transformer model.

> **Important:** This repository is a collection of independent experiments rather than one single end-to-end medical application. One notebook focuses on skin-lesion image classification, while the other notebooks explore weather time-series prediction, symptom-to-disease text classification, and news summarization.

## Experiments at a Glance

| Notebook | Task | Main model | Main input |
|---|---|---|---|
| `CNN.ipynb` | Skin-lesion image classification | Convolutional Neural Network | HAM10000 images and metadata |
| `RNN.ipynb` | Weather time-series classification | LSTM | Sequential weather measurements |
| `rnnodd.ipynb` | Symptom-to-disease text classification | Embedding + Bidirectional LSTM | Symptom descriptions |
| `transformer.ipynb` | News article summarization | T5-small Transformer | CNN/DailyMail article-summary pairs |

## Experiment 1: CNN for Skin-Lesion Classification

The CNN notebook uses the HAM10000 metadata file and a corresponding collection of skin images. The metadata contains an image identifier and a diagnostic label in the `dx` column. The notebook locates the image files, resizes them to `128 × 128`, normalizes pixel values to the `[0, 1]` range, and converts the diagnostic labels into one-hot encoded targets.

The model contains three convolution and max-pooling stages followed by a dense layer, dropout, and a softmax output layer. Training uses image augmentation with rotations, shifts, and flips. Early stopping and learning-rate reduction are used to make training more stable.

```text
HAM10000 metadata + image folders
              |
              v
Image lookup by image_id
              |
              v
Resize and pixel normalization
              |
              v
Train/test split + augmentation
              |
              v
CNN training
              |
              v
Test loss and accuracy
```

The notebook expects the image folders and the CSV metadata to be available locally. The raw images are not stored in this repository.

## Experiment 2: LSTM for Weather Time-Series Data

`RNN.ipynb` explores sequence modeling with an LSTM. The code reads weather data, encodes the precipitation type, scales numerical features with `MinMaxScaler`, and creates sequences of length 24. The intended target is the precipitation type, such as rain or snow.

The train/test split is performed without shuffling so that the temporal order is not destroyed. The LSTM receives a sequence of 24 time steps and returns a binary prediction.

> This notebook is a time-series experiment, not a medical-image model. Before running it, define the feature columns explicitly and verify that the selected CSV contains the expected weather fields.

## Experiment 3: Symptom-to-Disease Classification with BiLSTM

`rnnodd.ipynb` treats disease prediction as a text-classification problem. The input is a symptom description and the target is a condition label. The text is tokenized, converted into padded integer sequences, and passed through an embedding layer followed by a bidirectional LSTM and dense classification layers.

The notebook also provides a `predict_disease` helper that displays the top three predicted conditions with their model probabilities.

```text
Symptom text
    |
    v
Tokenizer and vocabulary
    |
    v
Padded token sequences
    |
    v
Embedding + Bidirectional LSTM
    |
    v
Disease-class probabilities
```

This is a machine-learning experiment and must not be used to diagnose real patients. The predictions are not medical advice and require clinical validation before any real-world use.

## Experiment 4: T5 Transformer for News Summarization

`transformer.ipynb` fine-tunes the `t5-small` sequence-to-sequence Transformer on CNN/DailyMail article and highlight pairs. The input is prefixed with `summarize:`, tokenized with maximum input and target lengths, and used to train a sequence-to-sequence model with the Hugging Face Trainer API.

After training, the notebook uses a summarization pipeline to generate summaries for sample articles.

```text
News article
    |
    v
T5 tokenizer
    |
    v
T5-small sequence-to-sequence model
    |
    v
Generated summary
```

The notebook expects `train.csv`, `validation.csv`, and `test.csv` under a local `cnn_dailymail/` directory with `article` and `highlights` columns.

## Repository Structure

```text
.
├── CNN.ipynb
├── RNN.ipynb
├── rnnodd.ipynb
├── transformer.ipynb
├── HAM10000_metadata.csv       # Kept in the separate dataset package
├── hmnist_28_28_RGB.csv        # Kept in the separate dataset package
├── hmnist_28_28_L.csv          # Kept in the separate dataset package
├── hmnist_8_8_RGB.csv          # Kept in the separate dataset package
├── hmnist_8_8_L.csv            # Kept in the separate dataset package
├── medical_dataset.csv         # Kept in the separate dataset package
├── weatherHistory.csv          # Kept in the separate dataset package
├── .gitignore
└── README.md
```

The CSV files are intentionally stored separately from GitHub because some of them are large. Download the project data archive from the Google Drive link provided by the project owner and place the required files in the expected local paths.

## Installation

Create a Python virtual environment:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the common dependencies:

```bash
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn pillow jupyter
pip install tensorflow transformers torch datasets
```

For GPU training, install a TensorFlow and PyTorch combination compatible with your operating system and CUDA version. A CPU installation is sufficient for small tests but can be slow for image training and Transformer fine-tuning.

## Run the Notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open one notebook at a time and update the local file paths before running its cells. The notebooks contain absolute Windows paths from the original development environment, so they will not work unchanged on another computer.

## CNN Run Instructions

1. Download or place the HAM10000 metadata file in the project data directory.
2. Place the corresponding image folders on the local machine.
3. Update `data_df = pd.read_csv(...)` in `CNN.ipynb`.
4. Update the two image-folder paths in the `folders` list.
5. Run the image lookup, preprocessing, augmentation, model-building, and training cells.
6. Review the test accuracy and loss.

The metadata `image_id` values must match the filenames in the image folders. Missing image files are skipped by the notebook and reported during loading.

## RNN Weather Run Instructions

1. Place the weather CSV in the project directory.
2. Update the path in `RNN.ipynb`.
3. Define the numerical feature list used to build `X`.
4. Confirm that `Precip Type` contains the expected binary classes.
5. Run scaling, sequence creation, LSTM training, and evaluation.

The current notebook contains an unfinished feature-selection step, so it may require a small code adjustment before it runs from top to bottom.

## Symptom-Classification Run Instructions

1. Place `medical_dataset.csv` in the expected path.
2. Confirm that it contains `symptoms` and `condition` columns.
3. Run the tokenizer and label-encoding cells.
4. Train the BiLSTM model.
5. Call `predict_disease("your symptom description")` for an experimental prediction.

This notebook is for educational classification experiments only. It is not a clinical system.

## Transformer Run Instructions

Prepare the following local structure:

```text
cnn_dailymail/
├── train.csv
├── validation.csv
└── test.csv
```

Each file should include:

```text
article,highlights
```

Then run the notebook cells in order. The first run downloads the tokenizer and `t5-small` model, so an internet connection and sufficient disk space are required.

## Limitations and Reproducibility

The notebooks are exploratory and do not currently share one unified configuration system. Dataset paths, model hyperparameters, and preprocessing settings are defined inside individual notebooks. For reproducible experiments, add a common configuration file, lock dependency versions, set random seeds, and save the train/validation/test split used by each experiment.

The medical-image notebook also depends on image files that are not included in the repository. The reported performance of any model depends on the exact images, labels, preprocessing, split strategy, and hardware used during training.

## Future Improvements

Recommended improvements include separating the four experiments into dedicated subdirectories, adding reusable Python modules, creating a clean requirements file for each experiment, adding automated evaluation reports, saving trained models and label encoders, implementing experiment tracking, and adding inference scripts for each task.

For the medical-image experiment, future work should include class balancing, patient-level splits, calibration, external validation, and a careful analysis of false positives and false negatives. For the disease-text experiment, any real-world use would require clinical review, privacy protection, and extensive validation.

## Responsible Use

The project includes medical-image and symptom-classification experiments. These examples are for education and research. They are not medical devices, diagnostic tools, or substitutes for qualified healthcare professionals. Do not use their predictions to make decisions about a person's health.

## License and Dataset Notice

Review the license and usage terms of each external dataset and pretrained model before redistribution. The GitHub repository contains the notebooks and source materials; large datasets should remain in their approved external storage location.
