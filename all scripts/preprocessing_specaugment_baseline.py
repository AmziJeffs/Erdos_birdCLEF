################################################################################
# SETUP
################################################################################

# Convenience and saving flags
ABRIDGED_RUN = False # Set to True to train and validate on 10% of the data, for quick funcitonality tests etc
SAVE_AFTER_TRAINING = True # Save the model when you are done
SAVE_CHECKPOINTS = True # Save the model after ever epoch
REPORT_TRAINING_LOSS_PER_EPOCH = True # Track the training loss each epoch, and write it to a file after training
REPORT_VALIDATION_LOSS_PER_EPOCH = True # Lets us make a nice learning curve after training

# Training hyperparameters
BATCH_SIZE = 256 # Number of samples per batch while training our network
NUM_EPOCHS = 20 # Number of epochs to train our network
LEARNING_RATE = 0.001 # Learning rate for our optimizer

# Directories
DATA_DIR = "data/"
AUDIO_DIR = DATA_DIR + "train_audio/"
CHECKPOINT_DIR = "../checkpoints/" # Checkpoints, models, and training data will be saved here
MODEL_NAME = None

# Preprocessing info
SAMPLE_RATE = 32000 # All our audio uses this sample rate
SAMPLE_LENGTH = 5 # Duration we want to crop our audio to
NUM_SPECIES = 182 # Number of bird species we need to label

################################################################################
# IMPORTS
################################################################################

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

################################################################################
# LOAD DATA
################################################################################

data = pd.read_csv(DATA_DIR+"train_metadata.csv")
data['filepath'] = AUDIO_DIR + data['filename']

# We only need the filepath and species label
data = data[['filepath', 'primary_label']]

# Replace string labels by tensors whose entries are dummies
species = data['primary_label'].unique()
species_to_index = {species[i]:i for i in range(len(species))}
data['tensor_label'] = pd.Series(pd.get_dummies(data['primary_label']).astype(int).values.tolist()).apply(lambda x: torch.Tensor(x))
data.sample(5)

# Train test split, stratified by species
data_train, data_test = train_test_split(data, test_size = 0.2, stratify=data['primary_label'])

# Use 10% of data for quick runs to test compile/functionality
if ABRIDGED_RUN == True:
    data_train = data_train.sample(int(len(data_train)*0.1))
    data_test = data_test.sample(int(len(data_train)*0.1))

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
    sample_rate=16000,  # Replace SAMPLE_RATE with actual value
    f_min=0,
    f_max=16000,
    n_stft=1025  # the number of frequency bins in the spectrogram
)

# Scales decibels to reasonable level (apply to a spectrogram or Mel spectrogram)
db_scaler = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

# Resizes spectrograms into square images
resize = transforms.Resize((224, 224), antialias = None)

# SpecAugment functions
def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    center = random.randrange(W, spec_len - W)
    warped = random.randint(center - W, center + W)
    left = torchaudio.functional.time_stretch(spec, 1, center / spec_len)
    right = torchaudio.functional.time_stretch(spec, 1, (spec_len - center) / spec_len)
    return torch.cat((left, right), dim=2)

def freq_mask(spec, F=30):
    num_mel_channels = spec.shape[1]
    f = random.randrange(0, F)
    f_zero = random.randrange(0, num_mel_channels - f)
    
    spec[:, f_zero:f_zero+f, :] = 0
    return spec

def time_mask(spec, T=40):
    spec_len = spec.shape[2]
    t = random.randrange(0, T)
    t_zero = random.randrange(0, spec_len - t)
    
    spec[:, :, t_zero:t_zero+t] = 0
    return spec

# Processes a sample to a tensor for our network, including SpecAugment
def sample_to_tensor(sample):
    x = spectrogram_transform(sample)
    x = mel_spectrogram_transform(x)
    
    # Apply SpecAugment
    x = time_warp(x)
    x = freq_mask(x)
    x = time_mask(x)
    
    x = db_scaler(x)
    x = resize(x)
    return x

# Takes a filepath and outputs a torch tensor with shape (1, 224, 224) 
# that we can feed into our CNN
def filepath_to_tensor(filepath):
    sample, _ = torchaudio.load(filepath)
    if len(sample) >= 16000 * 5:  # Replace SAMPLE_RATE * SAMPLE_LENGTH with actual values
        sample = sample[:16000 * 5]
    else:
        pad_length = 16000 * 5 - len(sample)
        sample = torch.nn.functional.pad(sample, (0, pad_length))
    return sample_to_tensor(sample)
################################################################################
# DATASET CLASS
################################################################################

# Note: filepaths and labels should be ordinary lists
class BirdDataset(Dataset):
    def __init__(self, filepaths, labels):
        super().__init__()
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        processed_clip = filepath_to_tensor(self.filepaths[index])
        return processed_clip, self.labels[index]

################################################################################
# ARCHITECTURE
################################################################################

class BirdClassifier(nn.Module):
    ''' Pared down architecture from https://github.com/musikalkemist/pytorchforaudio/blob/main/10%20Predictions%20with%20sound%20classifier/cnn.py'''
    def __init__(self, num_classes):
        super(BirdClassifier, self).__init__()
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
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.dropout = nn.Dropout(p=0.1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(25088, num_classes)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

################################################################################
# TRAINING SETUP
################################################################################

# Set device we'll train on
device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")

# Set model name if not set.
# Defaults to a timestamp, YYYY-MM-DD_HH_MM_SS
if MODEL_NAME == None:
    MODEL_NAME = str(pd.Timestamp.now()).replace(" ", "_").replace(":", "-").split(".")[0]

# Create a saving directory if needed
if SAVE_AFTER_TRAINING or SAVE_CHECKPOINTS or REPORT_TRAINING_LOSS_PER_EPOCH or REPORT_VALIDATION_LOSS_PER_EPOCH:
    output_dir = Path(f'{CHECKPOINT_DIR}{MODEL_NAME}')
    output_dir.mkdir(parents=True, exist_ok=True)

# Instantiate our training dataset
train_dataset = BirdDataset(filepaths = data_train['filepath'].to_list(), labels = data_train['tensor_label'].to_list())
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate our validation dataset
validation_dataset =  BirdDataset(filepaths = data_test['filepath'].to_list(), labels = data_test['tensor_label'].to_list())
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Instantiate our model
model = BirdClassifier(NUM_SPECIES).to(device)

# Set our loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Training loop
print(f"Training on {len(train_dataset)} samples with {BATCH_SIZE} samples per batch.")
if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
    print(f"Validating on {len(validation_dataset)} samples at the end of each epoch.")

training_losses = [None]*NUM_EPOCHS
validation_losses = [None]*NUM_EPOCHS

torch.enable_grad() # Turn on the gradient

################################################################################
# TRAINING LOOP
################################################################################

for epoch_num, epoch in enumerate(tqdm(range(NUM_EPOCHS))):

    running_loss = 0.0
    
    for i, data in enumerate(tqdm(train_dataloader, leave = False)):
        
        # Get batch of inputs and true labels, push to device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass on batch of inputs
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Update weights
        optimizer.step()

        # Update loss
        running_loss += loss.item()

    # Save checkpoint
    if SAVE_CHECKPOINTS == True:
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}{MODEL_NAME}/checkpoint_{epoch_num+1}.pt")

    # Compute training loss
    if REPORT_TRAINING_LOSS_PER_EPOCH == True:    
        training_losses[epoch_num] = running_loss/len(train_dataloader)
        
    # Compute validation loss
    if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
        validation_loss = 0.0
        model.eval()
        for validation_data in validation_dataloader:
            inputs, labels = validation_data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            validation_loss += criterion(outputs, labels).item()
        validation_losses[epoch_num] = validation_loss/len(validation_dataloader)
        model.train()

print('Finished Training')

################################################################################
# SAVE AND REPORT
################################################################################

if SAVE_AFTER_TRAINING == True:
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}{MODEL_NAME}/final.pt")

losses = pd.DataFrame({"training_losses":training_losses, "validation_losses":validation_losses})
cols = []
if REPORT_TRAINING_LOSS_PER_EPOCH == True:
    cols += ["training_losses"]
if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
    cols += ["validation_losses"]
if len(cols) > 0:
    losses[cols].to_csv(f"{CHECKPOINT_DIR}{MODEL_NAME}/losses.csv")
    print(losses)

