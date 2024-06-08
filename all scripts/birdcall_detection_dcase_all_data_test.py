################################################################################
# SETUP
################################################################################

# Convenience and saving flags
ABRIDGED_RUN = False  # Set to True to train and validate on 10% of the data, for quick funcitonality tests etc
SAVE_AFTER_TRAINING = True # Save the model when you are done

# Directories
DATA_DIR = "../data/"
AUDIO_DIR_DCASE = DATA_DIR + "wav/"
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
MODEL_NAME = "BIRDCALL_DETECTION_DCASE_ALL_DATA"

# Preprocessing info
SAMPLE_RATE = 44100 # All our audio uses this sample rate
SAMPLE_LENGTH = 5 # Duration we want to crop our audio to

################################################################################
# IMPORTS
################################################################################

print("Importing modules")

# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import librosa
import random
import os
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

# Torch imports
import torch
import torchaudio
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchaudio.transforms import MelSpectrogram, Resample
from IPython.display import Audio
import torch.optim as optim
from torcheval.metrics import BinaryAUROC


print("Done")

################################################################################
# LOAD DATA
################################################################################

dcase = pd.read_csv(DATA_DIR+'ff1010bird_metadata_2018.csv')

# Create a filepath column
dcase['filepath'] = AUDIO_DIR_DCASE + dcase['itemid'].astype(str)+'.wav'

print("Loading audio signals into memory")
tqdm.pandas()
def filepath_to_signal(filepath):
    sample, _ = torchaudio.load(filepath)
    return sample
dcase['signal'] = dcase['filepath'].progress_apply(filepath_to_signal)
print("Done")

# Train test split, stratified by 'hasbird'
dcase_train, dcase_test = train_test_split(dcase, test_size = 0.2, random_state=123, stratify=dcase['hasbird'])

if ABRIDGED_RUN == True:
    dcase_train = dcase_train.sample(int(len(dcase_train)*0.1))
    dcase_test = dcase_test.sample(int(len(dcase_train)*0.1))

################################################################################
# PREPROCESSING FUNCTIONS
################################################################################

# Transforms audio signal to a spectrogram
spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        power=2
    )

# Converts ordinary spectrogram to Mel scale
mel_spectrogram_transform = torchaudio.transforms.MelScale(
    n_mels=256,
    sample_rate=SAMPLE_RATE,
    f_min=0,
    f_max=16000,
    n_stft=1025  # the number of frequency bins in the spectrogram
)

# Scales decibels to reasonable level (apply to a spectrogram or Mel spectrogram)
db_scaler = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

# Resizes spectrograms into square images
resize = transforms.Resize((224, 224), antialias = None)

# Applies a frequency mask to a spectrogram
def freq_mask(spec, F=30):
    num_mel_channels = spec.shape[1]
    f = random.randrange(0, F)
    f_zero = random.randrange(0, num_mel_channels - f)
    spec[:, f_zero:f_zero+f, :] = 0
    return spec

# Applies a time mask to a spectrogram
def time_mask(spec, T=40):
    spec_len = spec.shape[2]
    t = random.randrange(0, T)
    t_zero = random.randrange(0, spec_len - t)
    spec[:, :, t_zero:t_zero+t] = 0
    return spec

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# DATASET CLASS
################################################################################

# Slice a sequence into segments
def slices(seq, window_size = SAMPLE_RATE*SAMPLE_LENGTH, stride = None, align_left = True, return_scraps = True):
    # If one window is larger than the sequence, just return the scraps or nothing
    if window_size > seq.squeeze().size(dim=0):
        if return_scraps == True:
            return [seq]
        else:
            return []
    # If stride is None, it defaults to window_size
    if stride == None:
        stride = window_size
    index_slices = []
    left_pointer = 0
    while left_pointer + window_size <= seq.squeeze().size(dim=0):
        index_slices += [[left_pointer, left_pointer + window_size]]
        left_pointer += stride
    if align_left == False:
        offset = seq.squeeze().size(dim=0)-(left_pointer-stride)-window_size
        index_slices = [[a+offset, b+offset] for [a,b] in index_slices]
    if return_scraps == True and left_pointer < seq.squeeze().size(dim=0):
        if align_left == True:
            index_slices += [[left_pointer, seq.squeeze().size(dim=0)]]
        else:
            index_slices += [[0, seq.squeeze().size(dim=0) - left_pointer]]
    return [seq[a:b] for [a,b] in index_slices]

# Note: signals and labels should be ordinary lists
# Training boolean is used to decide whether to apply masks
# Config should have the format of a dictionary
class DCaseData(Dataset):
    def __init__(self, signals, labels, training = True,
        config = {'use_mel': True, 'time_mask': True, 'freq_mask': True}):
        super().__init__()
        self.training = training
        self.config = config
        print(f'Preprocessing {"training" if training else "validation"} data\n')
        self.processed_clips, self.labels = self.process(signals, labels)
    def process(self, signals, labels):
        results = []
        new_labels = []
        for i, signal in enumerate(signals):
            # Uniformize to 5 seconds
            if signal.shape[1] < SAMPLE_RATE * SAMPLE_LENGTH:
                pad_length = SAMPLE_RATE * SAMPLE_LENGTH - signal.shape[1]
                signal = torch.nn.functional.pad(signal, (0, pad_length))
            # Cut signal into 5 second chunks to save
            for clip in slices(signal.squeeze()):
                results += [clip.unsqueeze(0)]
                new_labels += [labels[i]]
        return results, new_labels
    def __len__(self):
        return len(self.processed_clips)
    def __getitem__(self, index):
        # Process clip to tensor
        x = self.processed_clips[index]
        x = spectrogram_transform(x)
        if self.config['use_mel']:
            x = mel_spectrogram_transform(x)
        x = db_scaler(x)
        if self.config['time_mask'] and self.training:
            x = time_mask(x)
        if self.config['freq_mask'] and self.training:
            x = freq_mask(x)
        x = resize(x)
        return x, self.labels[index]
    
################################################################################
# ARCHITECTURE
################################################################################

class BirdCallDetector(nn.Module):
    ''' Full architecture from https://github.com/musikalkemist/pytorchforaudio/blob/main/10%20Predictions%20with%20sound%20classifier/cnn.py'''
    def __init__(self):
        super(BirdCallDetector, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(10368, 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
    
validation_dataset =  DCaseData(signals = dcase_test['signal'].to_list(), 
                                labels = dcase_test['hasbird'].to_list(),
                                training = False)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

################################################################################
# EVALUATE AUROC ON VALIDATION DATASET
################################################################################

metric = BinaryAUROC()
model = torch.load('BIRDCALL_DETECTION_DCASE_ALL_DATA.pt')



