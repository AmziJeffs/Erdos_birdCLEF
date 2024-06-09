################################################################################
# SETUP
################################################################################

# Convenience and saving flags
ABRIDGED_RUN = True
SAVE_AFTER_TRAINING = True # Save the model when you are done
SAVE_CHECKPOINTS = True # Save the model after ever epoch
REPORT_TRAINING_LOSS_PER_EPOCH = True # Track the training loss each epoch, and write it to a file after training
REPORT_VALIDATION_LOSS_PER_EPOCH = True # Lets us make a nice learning curve after training

# Training hyperparameters
BATCH_SIZE = 64 # Number of samples per batch while training our network
NUM_EPOCHS = 60 # Number of epochs to train our network
LEARNING_RATE = 0.001 # Learning rate for our optimizer

# Directories
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
DATA_DIR = "data/"
AUDIO_DIR = DATA_DIR + "train_audio/"

# We use the following pipeline (TODO)
# 1. Cut each clip into 5 second segments, with stride 2.5 seconds
# 2. Use a nocall detector to throw away the segments that don't
#    contain a bird call
# 3. When asking for a clip from the dataset, we
#    a. Make Mel spec
#    b. Apply random power 0.5 to 3
#    c. Frequency and time mask
# 4. We use weighted random sampling, by a count from each class
MODEL_NAME = "SIX_LAYERS_BCE_TEST"

# Preprocessing info
SAMPLE_RATE = 32000 # All our audio uses this sample rate
SAMPLE_LENGTH = 5 # Duration we want to crop our audio to
NUM_SPECIES = 182 # Number of bird species we need to label
MAX_SAMPLE_LENGTH = 60 # Trim every sample to <= 60 seconds

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
species = ['asbfly', 'ashdro1', 'ashpri1', 'ashwoo2', 'asikoe2', 'asiope1', 'aspfly1', 'aspswi1', 'barfly1', 'barswa', 'bcnher', 'bkcbul1', 'bkrfla1', 'bkskit1', 'bkwsti', 'bladro1', 'blaeag1', 'blakit1', 'blhori1', 'blnmon1', 'blrwar1', 'bncwoo3', 'brakit1', 'brasta1', 'brcful1', 'brfowl1', 'brnhao1', 'brnshr', 'brodro1', 'brwjac1', 'brwowl1', 'btbeat1', 'bwfshr1', 'categr', 'chbeat1', 'cohcuc1', 'comfla1', 'comgre', 'comior1', 'comkin1', 'commoo3', 'commyn', 'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crbsun2', 'cregos1', 'crfbar1', 'crseag1', 'dafbab1', 'darter2', 'eaywag1', 'emedov2', 'eucdov', 'eurbla2', 'eurcoo', 'forwag1', 'gargan', 'gloibi', 'goflea1', 'graher1', 'grbeat1', 'grecou1', 'greegr', 'grefla1', 'grehor1', 'grejun2', 'grenig1', 'grewar3', 'grnsan', 'grnwar1', 'grtdro1', 'gryfra', 'grynig2', 'grywag', 'gybpri1', 'gyhcaf1', 'heswoo1', 'hoopoe', 'houcro1', 'houspa', 'inbrob1', 'indpit1', 'indrob1', 'indrol2', 'indtit1', 'ingori1', 'inpher1', 'insbab1', 'insowl1', 'integr', 'isbduc1', 'jerbus2', 'junbab2', 'junmyn1', 'junowl1', 'kenplo1', 'kerlau2', 'labcro1', 'laudov1', 'lblwar1', 'lesyel1', 'lewduc1', 'lirplo', 'litegr', 'litgre1', 'litspi1', 'litswi1', 'lobsun2', 'maghor2', 'malpar1', 'maltro1', 'malwoo1', 'marsan', 'mawthr1', 'moipig1', 'nilfly2', 'niwpig1', 'nutman', 'orihob2', 'oripip1', 'pabflo1', 'paisto1', 'piebus1', 'piekin1', 'placuc3', 'plaflo1', 'plapri1', 'plhpar1', 'pomgrp2', 'purher1', 'pursun3', 'pursun4', 'purswa3', 'putbab1', 'redspu1', 'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar', 'rossta2', 'rufbab3', 'ruftre2', 'rufwoo2', 'rutfly6', 'sbeowl1', 'scamin3', 'shikra1', 'smamin1', 'sohmyn1', 'spepic1', 'spodov', 'spoowl1', 'sqtbul1', 'stbkin1', 'sttwoo1', 'thbwar1', 'tibfly3', 'tilwar1', 'vefnut1', 'vehpar1', 'wbbfly1', 'wemhar1', 'whbbul2', 'whbsho3', 'whbtre1', 'whbwag1', 'whbwat1', 'whbwoo2', 'whcbar1', 'whiter2', 'whrmun', 'whtkin2', 'woosan', 'wynlau1', 'yebbab1', 'yebbul3', 'zitcis1']
species_to_index = {species[i]:i for i in range(len(species))}
data['index_label'] = data['primary_label'].apply(lambda x: species_to_index[x])
data['tensor_label'] = pd.Series(pd.get_dummies(data['primary_label']).astype(int).values.tolist()).apply(lambda x: torch.Tensor(x))
data.sample(5)

# Use 100 rows of data for quick runs to test functionalities
if ABRIDGED_RUN == True:
    data = data.sample(10)

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
                pad_length = SAMPLE_RATE * SAMPLE_LENGTH - signal.shape[1]
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
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(46208, num_classes)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

################################################################################
# WEIGHTED RANDOM SAMPLER SETUP
################################################################################

label_counts = data_train['primary_label'].value_counts() # TODO, count from dataset
weights = data_train['primary_label'].apply(lambda x: 1/label_counts.loc[x])
weighted_sampler = torch.utils.data.WeightedRandomSampler(weights.to_list(), len(weights))

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
train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler = weighted_sampler)

# Instantiate our validation dataset
validation_dataset =  BirdDataset(signals = data_validation['signal'].to_list(), 
                                  labels = data_validation['tensor_label'].to_list(),
                                  training = False)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Instantiate our model
model = BirdClassifier(NUM_SPECIES).to(device)

# Set our loss function and optimizer
criterion = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Training loop
print(f"Training on {len(train_dataset)} samples with {BATCH_SIZE} samples per batch.")
if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
    print(f"Validating on {len(validation_dataset)} samples at the end of each epoch.")

training_losses = [None]*NUM_EPOCHS
validation_losses = [None]*NUM_EPOCHS

torch.enable_grad() # Turn on the gradient

################################################################################
# SAVE AND REPORT train test split
################################################################################

# Save train test split
data_train[['primary_label', 'filepath']].to_csv(f"{output_dir}/data_train.csv", index = False)
data_validation[['primary_label', 'filepath']].to_csv(f"{output_dir}/data_validation.csv", index = False)

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
        print(loss.item())
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
        torch.no_grad()
        for validation_data in validation_dataloader:
            inputs, labels = validation_data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            validation_loss += criterion(outputs, labels).item()
        validation_losses[epoch_num] = validation_loss/len(validation_dataloader)
        model.train()
        torch.enable_grad()

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