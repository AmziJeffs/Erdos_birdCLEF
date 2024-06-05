#!/usr/bin/env python
# coding: utf-8

# # 0. Setup

# In[1]:


# Convenience and saving
ABRIDGED_RUN = False # Set to True to train and validate on 10% of the data, for quick funcitonality tests etc
SAVE_AFTER_TRAINING = True # Save the model when you are done
SAVE_CHECKPOINTS = True # Save the model after every epoch
REPORT_TRAINING_LOSS_PER_EPOCH = True # Track the training loss each epoch, and write it to a file after training
REPORT_VALIDATION_LOSS_PER_EPOCH = True # Lets us make a nice learning curve after training

# Training hyperparameters
BATCH_SIZE = 32 # Number of samples per batch while training our network
NUM_EPOCHS = 20 # Number of epochs to train our network
LEARNING_RATE = 0.001 # Learning rate for our optimizer

# Directories
DATA_DIR = "data/"
AUDIO_DIR_2021 = DATA_DIR + "train_soundscapes_2021/"
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
MODEL_NAME = None

# Preprocessing info
SAMPLE_RATE = 32000 # All our audio uses this sample rate
SAMPLE_LENGTH = 5 # Duration we want to crop our audio to


# In[2]:


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
from pathlib import Path


# In[5]:


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


# In[6]:


sounds = pd.read_csv(DATA_DIR+'train_soundscape_2021_labels.csv')


# In[7]:


sounds.head()


# In[8]:


def get_filepath(row):
    filepath = AUDIO_DIR_2021+ str(row.audio_id)+'_'+row.site+'.ogg'
    return filepath


# In[9]:


sounds['filepath'] = sounds.apply(get_filepath, axis=1)

################################################################################
tqdm.pandas()
# Takes a row_id and loads the signal to memory and returns it
def row_id_to_signal(row_id):
    row_id_parts = row_id.split('_')
    filepath = AUDIO_DIR_2021+ row_id_parts[0]+'_'+row_id_parts[1]+'.ogg'
    seconds = int(row_id_parts[2])
    sample, _ = torchaudio.load(filepath)
    end = seconds*SAMPLE_RATE
    start = (seconds-5)*SAMPLE_RATE
    sample = sample[0:1, start:end]
    return sample
print("Loading signals to memory")
sounds['signal'] = sounds['row_id'].progress_apply(row_id_to_signal)

################################################################################

# In[10]:


sounds.head()


# In[11]:


sounds['has_call'] = 1*(sounds.birds != 'nocall')


# In[12]:


sounds.head()


# In[13]:


sounds.has_call.describe()


# In[14]:


sounds_train, sounds_test = train_test_split(sounds, test_size = 0.2, random_state=123, stratify=sounds['has_call'])

if ABRIDGED_RUN == True:
    sounds_train = sounds_train.sample(int(len(sounds_train)*0.1))
    sounds_test = sounds_test.sample(int(len(sounds_train)*0.1))

print(sounds_train.info())


# # 2. Preprocessing and Generating Spectrograms

# In[15]:


# Transforms audio signal to a spectrogram
spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        power=2
    )


# In[16]:


# Converts ordinary spectrogram to Mel scale
mel_spectrogram_transform = torchaudio.transforms.MelScale(
    n_mels=256,
    sample_rate=SAMPLE_RATE,
    f_min=0,
    f_max=16000,
    n_stft=1025  # the number of frequency bins in the spectrogram
)


# In[17]:


# Scales decibels to reasonable level (apply to a spectrogram or Mel spectrogram)
db_scaler = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)


# In[18]:


# Resizes spectrograms into square images
resize = transforms.Resize((224, 224), antialias = None)


# In[19]:


# Processes a sample to a tensor for our network
def sample_to_tensor(sample):
    x = spectrogram_transform(sample)
    x = mel_spectrogram_transform(x)
    x = db_scaler(x)
    x = resize(x)
    return x


# In[20]:

################################################################################
# Takes a row and outputs a torch tensor with shape (1, 224, 224) 
# that we can feed into our CNN
def row_to_tensor(row_id):
    sample = sounds[sounds['row_id'] == row_id]['signal'].iloc[0]
    return sample_to_tensor(sample)
################################################################################


# In[21]:


# Visualize a random spectrogram
row_index_sample = np.random.randint(0, len(sounds_train))
t = row_to_tensor(sounds_train.loc[row_index_sample, 'row_id'])
#plt.imshow(t.squeeze().numpy(), cmap='gray')
#plt.show()


# In[22]:


sounds_train.head()


# # 3. Set up torch dataset

# In[23]:


# Note: filepaths and labels should be ordinary lists
class BirdDataset2021(Dataset):
    def __init__(self, row_ids, labels):
        super().__init__()
        self.row_ids = row_ids
        self.labels = labels

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, index):
        processed_clip = row_to_tensor(self.row_ids[index])
        return processed_clip, self.labels[index]


# # 4. Defining neural network architecture 

# In[24]:


class BirdCallIdentifier(nn.Module):
    ''' Pared down architecture from https://github.com/musikalkemist/pytorchforaudio/blob/main/10%20Predictions%20with%20sound%20classifier/cnn.py'''
    def __init__(self, num_classes):
        super(BirdCallIdentifier, self).__init__()
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


# # 6. Training

# In[25]:


# Set device we'll train on
device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")


# In[26]:


# Set model name if not set.
# Defaults to a timestamp, YYYY-MM-DD_HH_MM_SS
if MODEL_NAME == None:
    MODEL_NAME = 'birdcall_detection_'+str(pd.Timestamp.now()).replace(" ", "_").replace(":", "-").split(".")[0]

# Create a saving directory if needed
if SAVE_AFTER_TRAINING or SAVE_CHECKPOINTS or REPORT_TRAINING_LOSS_PER_EPOCH or REPORT_VALIDATION_LOSS_PER_EPOCH:
    output_dir = Path(f'{CHECKPOINT_DIR}{MODEL_NAME}')
    output_dir.mkdir(parents=True, exist_ok=True)


# In[27]:


# Instantiate our training dataset
train_dataset = BirdDataset2021(row_ids = sounds_train['row_id'].to_list(), labels = sounds_train['has_call'].to_list())
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# In[28]:


# Instantiate our validation dataset
validation_dataset =  BirdDataset2021(row_ids = sounds_test['row_id'].to_list(), labels = sounds_test['has_call'].to_list())
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)


# In[29]:


# Instantiate our model
model = BirdCallIdentifier(2).to(device)


# In[30]:


# Set our loss function and optimizer
criterion = torcheval.metrics.BinaryAccuracy()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


# In[31]:


# Training loop
print(f"Training on {len(train_dataset)} samples with {BATCH_SIZE} samples per batch.")
if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
    print(f"Validating on {len(validation_dataset)} samples at the end of each epoch.")

training_losses = [None]*NUM_EPOCHS
validation_losses = [None]*NUM_EPOCHS

torch.enable_grad() # Turn on the gradient

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


# # 6. Save the model and report results

# In[193]:


if SAVE_AFTER_TRAINING == True:
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}{MODEL_NAME}/final.pt")


# In[194]:


losses = pd.DataFrame({"training_losses":training_losses, "validation_losses":validation_losses})
cols = []
if REPORT_TRAINING_LOSS_PER_EPOCH == True:
    cols += ["training_losses"]
if REPORT_VALIDATION_LOSS_PER_EPOCH == True:
    cols += ["validation_losses"]
if len(cols) > 0:
    losses[cols].to_csv(f"{CHECKPOINT_DIR}{MODEL_NAME}/losses.csv")
    print(losses)


# In[ ]:




