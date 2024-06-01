################################################################################
# SETUP
################################################################################

# Convenience and saving flags
ABRIDGED_RUN = False
SAVE_AFTER_TRAINING = True # Save the model when you are done
SAVE_CHECKPOINTS = True # Save the model after ever epoch
REPORT_TRAINING_LOSS_PER_EPOCH = True # Track the training loss each epoch, and write it to a file after training
REPORT_VALIDATION_LOSS_PER_EPOCH = True # Lets us make a nice learning curve after training

# Training hyperparameters
BATCH_SIZE = 256 # Number of samples per batch while training our network
NUM_EPOCHS = 20 # Number of epochs to train our network
LEARNING_RATE = 0.001 # Learning rate for our optimizer

# Directories
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
DATA_DIR = "data/"
AUDIO_DIR = DATA_DIR + "train_audio/"
MODEL_NAME = None

# Preprocessing info
SAMPLE_RATE = 32000 # All our audio uses this sample rate
SAMPLE_LENGTH = 5 # Duration we want to crop our audio to
NUM_SPECIES = 182 # Number of bird species we need to label

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

print("Done")

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

# Use 100 rows of data for quick runs to test functionalities
if ABRIDGED_RUN == True:
    data = data.sample(100)

print("Loading audio signals into memory")
tqdm.pandas()
def filepath_to_signal(filepath):
    sample, _ = torchaudio.load(filepath)
    return sample
data['signal'] = data['filepath'].progress_apply(filepath_to_signal)
print("Done")

# Train test split, stratified by species
stratify = data['primary_label']
if ABRIDGED_RUN == True:
    stratify = None
data_train, data_validation = train_test_split(data, test_size = 0.2, stratify=stratify)


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

MODEL_NAME = "FOUR_LAYERS_ALL_DATA"

################################################################################
# DATASET CLASS
################################################################################

# Slice a sequence into segments
def slices(seq, window_size = 160000, stride = None, align_left = True, return_scraps = False):
    # If one window is larger than the sequence, just return the scraps or nothing
    if window_size > len(seq):
        if return_scraps == True:
            return [seq]
        else:
            return []

    # If stride is None, it defaults to window_size
    if stride == None:
        stride = window_size

    index_slices = []
    left_pointer = 0
    while left_pointer + window_size <= len(seq):
        index_slices += [[left_pointer, left_pointer + window_size]]
        left_pointer += stride

    if align_left == False:
        offset = len(seq)-(left_pointer-stride)-window_size
        index_slices = [[a+offset, b+offset] for [a,b] in index_slices]

    if return_scraps == True and left_pointer + window_size > len(seq):
        if align_left == True:
            index_slices += [[left_pointer, len(seq)]]
        else:
            index_slices += [[0, len(seq) - left_pointer]]

    return [seq[a:b] for [a,b] in index_slices]

# Note: signals and labels should be ordinary lists
# Training boolean is used to decide whether to apply masks
# Config should have the format of a dictionary
class BirdDataset(Dataset):
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
        for i, signal in enumerate(tqdm(signals, total = len(signals), leave = False)):
            # Uniformize to 5 seconds
            if signal.shape[1] < SAMPLE_RATE * SAMPLE_LENGTH:
                pad_length = SAMPLE_RATE * SAMPLE_LENGTH - len(signal)
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

class BirdClassifier(nn.Module):
    ''' Full architecture from https://github.com/musikalkemist/pytorchforaudio/blob/main/10%20Predictions%20with%20sound%20classifier/cnn.py'''
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
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28800, num_classes)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
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
print(f"Using {device} for training")

# Create a saving directory if needed
output_dir = Path(f'{CHECKPOINT_DIR}{MODEL_NAME}')
output_dir.mkdir(parents=True, exist_ok=True)
output_dir = f'{CHECKPOINT_DIR}{MODEL_NAME}'

# Instantiate our training dataset
train_dataset = BirdDataset(signals = data_train['signal'].to_list(), 
                            labels = data_train['tensor_label'].to_list(),
                            training = True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate our validation dataset
validation_dataset =  BirdDataset(signals = data_validation['signal'].to_list(), 
                                  labels = data_validation['tensor_label'].to_list(),
                                  training = False)
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

for epoch_num, epoch in enumerate(tqdm(range(NUM_EPOCHS), leave = False)):

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

# Save train test split
data_train[['primary_label', 'filepath']].to_csv(f"{CHECKPOINT_DIR}{MODEL_NAME}/data_train.csv", index = False)
data_validation[['primary_label', 'filepath']].to_csv(f"{CHECKPOINT_DIR}{MODEL_NAME}/data_validation.csv", index = False)

# Save model
if SAVE_AFTER_TRAINING == True:
    torch.save(model.state_dict(), f'{output_dir}/final.pt')

# Save losses
losses = pd.DataFrame({"training_losses":training_losses, "validation_losses":validation_losses})
cols = []
if REPORT_TRAINING_LOSS_PER_EPOCH == True:
    cols += ["training_losses"]
if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
    cols += ["validation_losses"]
if len(cols) > 0:
    losses[cols].to_csv(f'{output_dir}/losses.csv', index = False)

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