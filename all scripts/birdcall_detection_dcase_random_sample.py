################################################################################
# SETUP
################################################################################

# Convenience and saving flags
ABRIDGED_RUN = False # Set to True to train and validate on 10% of the data, for quick funcitonality tests etc
SAVE_AFTER_TRAINING = True # Save the model when you are done
SAVE_CHECKPOINTS = True # Save the model after every epoch
REPORT_TRAINING_LOSS_PER_EPOCH = True # Track the training loss each epoch, and write it to a file after training
REPORT_VALIDATION_LOSS_PER_EPOCH = True # Lets us make a nice learning curve after training

# Training hyperparameters
BATCH_SIZE = 256 # Number of samples per batch while training our network
NUM_EPOCHS = 20 # Number of epochs to train our network
LEARNING_RATE = 0.001 # Learning rate for our optimizer

# Directories
DATA_DIR = "../data/"
AUDIO_DIR_DCASE = DATA_DIR + "wav/"
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
MODEL_NAME = "BIRDCALL_DETECTION_DCASE_RANDOM_SAMPLE"

# Preprocessing info
SAMPLE_RATE = 32000 # All our audio uses this sample rate
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
import os
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from pathlib import Path
import random

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
        self.labels = labels
        self.processed_clips, self.labels = self.process(signals)
        
    def process(self, signals):
        results = []
        for i, signal in enumerate(tqdm(signals, total = len(signals), leave = False)):
            # Uniformize to at least 5 seconds
            if signal.shape[1] < SAMPLE_RATE * SAMPLE_LENGTH:
                pad_length = SAMPLE_RATE * SAMPLE_LENGTH - len(signal)
                signal = torch.nn.functional.pad(signal, (0, pad_length))
            results += [signal]
        return results

    def __len__(self):
        return len(self.processed_clips)

    def __getitem__(self, index):
        # Process clip to tensor
        x = self.processed_clips[index]

        # Get a random 5-second clip from the whole sample
        start = np.random.randint(x.shape[1]-SAMPLE_RATE*SAMPLE_LENGTH+1)
        x = x[:, start:start + SAMPLE_RATE*SAMPLE_LENGTH]

        # Process
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
    
################################################################################
# TRAINING SETUP
################################################################################

# Set device we'll train on
device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set model name if not set.
# Defaults to a timestamp, YYYY-MM-DD_HH_MM_SS
if MODEL_NAME == None:
    MODEL_NAME = 'birdcall_detection_'+str(pd.Timestamp.now()).replace(" ", "_").replace(":", "-").split(".")[0]

# Create a saving directory if needed
if SAVE_AFTER_TRAINING or SAVE_CHECKPOINTS or REPORT_TRAINING_LOSS_PER_EPOCH or REPORT_VALIDATION_LOSS_PER_EPOCH:
    output_dir = Path(f'{CHECKPOINT_DIR}{MODEL_NAME}')
    output_dir.mkdir(parents=True, exist_ok=True)

# Instantiate our training dataset
train_dataset = DCaseData(signals = dcase_train['signal'].to_list(), 
                                labels = dcase_train['hasbird'].to_list(),
                                training=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate our validation dataset
validation_dataset =  DCaseData(signals = dcase_test['signal'].to_list(), 
                                labels = dcase_test['hasbird'].to_list(),
                                training = False)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Instantiate our model
model = BirdCallDetector().to(device)

# Set our loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

################################################################################
# TRAINING LOOP
################################################################################

print(f"Training on {len(train_dataset)} samples with {BATCH_SIZE} samples per batch.")
if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
    print(f"Validating on {len(validation_dataset)} samples at the end of each epoch.")

training_losses = [None]*NUM_EPOCHS
validation_losses = [None]*NUM_EPOCHS

torch.enable_grad() # Turn on the gradient

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

if SAVE_AFTER_TRAINING == True:
    torch.save(model.state_dict(), f'{output_dir}/final.pt')