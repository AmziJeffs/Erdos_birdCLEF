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
from tqdm import tqdm
print("Basic imports done.")



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
print("Torch imports done.")



# Directories
DATA_DIR = "data/"
AUDIO_DIR = DATA_DIR + "train_audio/"

# Preprocessing info
SAMPLE_RATE = 32000 # All our audio uses this sample rate
SAMPLE_LENGTH = 5 # Duration we want to crop our audio to
NUM_SPECIES = 182 # Number of bird species we need to label

# Training hyperparameters
BATCH_SIZE = 256 # Number of samples per batch while training our network
NUM_EPOCHS = 50 # Number of epochs to train our network
LEARNING_RATE = 0.001 # Learning rate for our optimizer

# Convenience and saving
ABRIDGED_TRAINING = False # Set to True to train on 10% of the training data, for quick funcitonality tests etc
SAVE_AFTER_TRAINING = True




# Load data
data = pd.read_csv(DATA_DIR+"train_metadata.csv")
data['filepath'] = AUDIO_DIR + data['filename']

# We only need the filepath and species label
data = data[['filepath', 'primary_label']]

# Replace string labels by tensors whose entries are dummies
species = data['primary_label'].unique()
species_to_index = {species[i]:i for i in range(len(species))}
data['tensor_label'] = pd.Series(pd.get_dummies(data['primary_label']).astype(int).values.tolist()).apply(lambda x: torch.Tensor(x))
print("Loaded data")



# Train test split, stratified by species
data_train, data_test = train_test_split(data, test_size = 0.2, stratify=data['primary_label'])

if ABRIDGED_TRAINING == True:
    data_train = data_train.sample(int(len(data_train)*0.1))

print(data_train.info())




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

# Takes a filepath and outputs a torch tensor with shape (1, 224, 224) 
# that we can feed into our CNN
def filepath_to_tensor(filepath):
    sample, _ = torchaudio.load(filepath)
    if len(sample) >= SAMPLE_RATE * SAMPLE_LENGTH:
        sample = sample[:SAMPLE_RATE * SAMPLE_LENGTH]
    else:
        pad_length = SAMPLE_RATE * SAMPLE_LENGTH - len(sample)
        sample = torch.nn.functional.pad(sample, (0, pad_length))
    spec = spectrogram_transform(sample)
    mel_spec = mel_spectrogram_transform(spec)
    db_scaled_mel_spec = db_scaler(mel_spec)
    resized = resize(db_scaled_mel_spec)
    return resized





# Note: filepaths and labels should be ordinary lists
class birdCLEF_dataset(Dataset):
    def __init__(self, filepaths, labels):
        super().__init__()
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        processed_clip = filepath_to_tensor(self.filepaths[index])
        return processed_clip, self.labels[index]




class birdClassifier(nn.Module):
    ''' Pared down architecture from https://github.com/musikalkemist/pytorchforaudio/blob/main/10%20Predictions%20with%20sound%20classifier/cnn.py'''
    def __init__(self, num_classes):
        super(birdClassifier, self).__init__()
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
        logits = self.linear(x)
        return logits






 # Set device we'll train on
device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Training with {device}")

# Instantiate our dataset
train_dataset = birdCLEF_dataset(filepaths = data_train['filepath'].to_list(), labels = data_train['tensor_label'].to_list())
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate our model
model = birdClassifier(NUM_SPECIES).to(device)


# Set our loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Training loop
print_per_n_batches = 10
print(f"Training on {len(train_dataset)} samples with {BATCH_SIZE} samples per batch.")

torch.enable_grad() # Turn on the gradient

training_losses = [0]*NUM_EPOCHS

for epoch in tqdm(range(NUM_EPOCHS)):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader, leave=False)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Update weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_per_n_batches == print_per_n_batches-1:    # print after some batches
            print(f'[{epoch + 1}/{NUM_EPOCHS}, {(i + 1):5d}/{len(train_dataloader)}] loss: {running_loss / print_per_n_batches:.3f}')
            running_loss = 0.0
    training_losses[epoch] = running_loss / len(train_dataloader)
print('Finished Training')
print(training_losses)






if SAVE_AFTER_TRAINING == True:
    torch.save(model.state_dict(), "".join(str(pd.Timestamp.now()).split())+"_NUM_EPOCHS_"+str(NUM_EPOCHS)+".pt")


print("Starting validation")
test_dataset =  birdCLEF_dataset(filepaths = data_test['filepath'].to_list(), labels = data_test['tensor_label'].to_list())
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

torch.no_grad() # Turn off gradient for fast validation loss computations
model.eval()

losses = []
for i, data in enumerate(test_dataloader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    pred = model(inputs)
    losses += [float(criterion(pred, labels))]
losses = pd.Series(losses)
losses.index = data_test.index
data_test['loss'] = losses
print(f"Average loss on test data: {data_test['loss'].sum() / len(data_test['loss'])}")
