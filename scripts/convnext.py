################################################################################
# SETUP
################################################################################

# Convenience and saving flags
#ABRIDGED_RUN = True
ABRIDGED_RUN = False
SAVE_AFTER_TRAINING = True # Save the model when you are done
SAVE_CHECKPOINTS = True # Save the model after ever epoch
REPORT_TRAINING_LOSS_PER_EPOCH = True # Track the training loss each epoch, and write it to a file after training
REPORT_VALIDATION_LOSS_PER_EPOCH = True # Lets us make a nice learning curve after training

# Training hyperparameters
BATCH_SIZE = 16 # Number of samples per batch while training our network
NUM_EPOCHS = 60 # Number of epochs to train our network
LEARNING_RATE = 0.0001 # Learning rate for our optimizer

# Directories
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
DATA_DIR = "../data/"
AUDIO_DIR = DATA_DIR + "train_audio/"
MODEL_NAME = None

# Preprocessing info
SAMPLE_RATE = 32000 # All our audio uses this sample rate
SAMPLE_LENGTH = 5 # Duration we want to crop our audio to
NUM_SPECIES = 182 # Number of bird species we need to label
MAX_SAMPLE_LENGTH = 60 # Trim every sample to <= 60 seconds
MIN_SAMPLE_LENGTH_NR = 10

IMAGE_LENGTH = 256
NUM_WORKERS = 16
################################################################################
# IMPORTS
################################################################################

# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import random
import os
import IPython.display as ipd
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torchvision.models import convnext
from torchgating import TorchGating as TG

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

# Use 100 rows of data for quick runs to test functionalities
if ABRIDGED_RUN == True:
    data = data.sample(20)

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
resize = transforms.Resize((IMAGE_LENGTH, IMAGE_LENGTH), antialias = None)

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

tg = TG(sr=SAMPLE_RATE, nonstationary=True)

################################################################################
# DATASET CLASS
################################################################################

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
            # Uniformize to at least 5 seconds
            if signal.shape[1] < SAMPLE_RATE * SAMPLE_LENGTH:
                pad_length = SAMPLE_RATE * SAMPLE_LENGTH - len(signal)
                signal = torch.nn.functional.pad(signal, (0, pad_length))
            results += [signal]
            new_labels += [labels[i]]
        return results, new_labels

    def __len__(self):
        return len(self.processed_clips)

    def __getitem__(self, index):
        # Process clip to tensor
        x = self.processed_clips[index]

        # Get a random 5-second clip from the whole sample
        start = np.random.randint(x.shape[1]-SAMPLE_RATE*SAMPLE_LENGTH+1)
        x = x[:, start:start + SAMPLE_RATE*SAMPLE_LENGTH]

        # Process
        x = tg(x)
        x = spectrogram_transform(x)
        if self.config['use_mel']:
            x = mel_spectrogram_transform(x)
        if self.training:
            exponent = np.random.uniform(low = 0.5, high = 3)
            x = torch.pow(x, exponent)
        x = db_scaler(x)
        if self.config['time_mask'] and self.training:
            x = time_mask(x)
        if self.config['freq_mask'] and self.training:
            x = freq_mask(x)
        x = resize(x)
        return x.expand(3, IMAGE_LENGTH, IMAGE_LENGTH), self.labels[index]
    
    
################################################################################
# TRAINING SETUP
################################################################################

# Set device we'll train on
device = None
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
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
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)

# Instantiate our validation dataset
validation_dataset =  BirdDataset(signals = data_validation['signal'].to_list(), 
                                  labels = data_validation['tensor_label'].to_list(),
                                  training = False)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers = NUM_WORKERS)

# Instantiate our model
#model = models.convnext_base(weights = 'DEFAULT')
model = models.convnext_base(weights = 'DEFAULT')
model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_SPECIES, bias=True, dtype=torch.float32)
model.to(device)

# Set our loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False)


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
    print(f"Epoch {epoch_num}")
    
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
        running_loss += float(loss.item())

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
            validation_loss += float(criterion(outputs, labels).item())
        validation_losses[epoch_num] = validation_loss/len(validation_dataloader)
        model.train()

    # Save losses
    losses = pd.DataFrame({"training_losses":training_losses, "validation_losses":validation_losses})
    cols = []
    if REPORT_TRAINING_LOSS_PER_EPOCH == True:
        cols += ["training_losses"]
    if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
        cols += ["validation_losses"]
    if len(cols) > 0:
        losses[cols].to_csv(f'{output_dir}/losses.csv', index = False)

print('Finished Training')

################################################################################
# SAVE AND REPORT 
################################################################################

# Save model
if SAVE_AFTER_TRAINING == True:
    torch.save(model.state_dict(), f'{output_dir}/final.pt')

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