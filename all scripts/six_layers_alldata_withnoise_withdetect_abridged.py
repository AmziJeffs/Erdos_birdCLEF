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
BATCH_SIZE = 32 # Number of samples per batch while training our network
NUM_EPOCHS = 60 # Number of epochs to train our network
LEARNING_RATE = 0.001 # Learning rate for our optimizer

# Directories
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
DATA_DIR = "../data/"
AUDIO_DIR = DATA_DIR + "train_audio/"

# CURRENT PIPELINE:
#  - Split each audio file into 5 second clips
#  - Discard any scrap with duration less than 3.5s. Pad others.
#  - Run Birdcall detection on each clip and change labels appropriately
#  - Loss function is BCEWithLogitsLoss
#  - Train with freq/time masking, random power, and pink bg noise.
#  - Validate without freq/time masking, random power, or bg noise.
#  - Model output will be a vector of logits. Need to apply sigmoid to get probabilities.
MODEL_NAME = "BCE_ALLDATA_PINKBG_WITHDETECT_ABRIDGED"
DETECTION_MODEL = 'BIRDCALL_DETECTION_DCASE_FINAL'

# Preprocessing info
SAMPLE_RATE = 32000 # All our audio uses this sample rate
SAMPLE_LENGTH = 5 # Duration we want to crop our audio to
NUM_SPECIES = 182 # Number of bird species we need to label
MIN_SAMPLE_LENGTH = 3.5 # Only use samples with length >= 3.5 seconds

# Min and max signal to noise ratio
MAX_SNR = 20
MIN_SNR = 10

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

data = pd.read_csv(DATA_DIR+"full_metadata.csv")
data['filepath'] = AUDIO_DIR + data['filename']

# We only need the filepath, primary label, and duration(?)
data = data[['filepath', 'primary_label', 'duration']]

# Replace string labels by tensors whose entries are dummies
species = ['asbfly', 'ashdro1', 'ashpri1', 'ashwoo2', 'asikoe2', 'asiope1', 'aspfly1', 'aspswi1', 'barfly1', 'barswa', 'bcnher', 'bkcbul1', 'bkrfla1', 'bkskit1', 'bkwsti', 'bladro1', 'blaeag1', 'blakit1', 'blhori1', 'blnmon1', 'blrwar1', 'bncwoo3', 'brakit1', 'brasta1', 'brcful1', 'brfowl1', 'brnhao1', 'brnshr', 'brodro1', 'brwjac1', 'brwowl1', 'btbeat1', 'bwfshr1', 'categr', 'chbeat1', 'cohcuc1', 'comfla1', 'comgre', 'comior1', 'comkin1', 'commoo3', 'commyn', 'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crbsun2', 'cregos1', 'crfbar1', 'crseag1', 'dafbab1', 'darter2', 'eaywag1', 'emedov2', 'eucdov', 'eurbla2', 'eurcoo', 'forwag1', 'gargan', 'gloibi', 'goflea1', 'graher1', 'grbeat1', 'grecou1', 'greegr', 'grefla1', 'grehor1', 'grejun2', 'grenig1', 'grewar3', 'grnsan', 'grnwar1', 'grtdro1', 'gryfra', 'grynig2', 'grywag', 'gybpri1', 'gyhcaf1', 'heswoo1', 'hoopoe', 'houcro1', 'houspa', 'inbrob1', 'indpit1', 'indrob1', 'indrol2', 'indtit1', 'ingori1', 'inpher1', 'insbab1', 'insowl1', 'integr', 'isbduc1', 'jerbus2', 'junbab2', 'junmyn1', 'junowl1', 'kenplo1', 'kerlau2', 'labcro1', 'laudov1', 'lblwar1', 'lesyel1', 'lewduc1', 'lirplo', 'litegr', 'litgre1', 'litspi1', 'litswi1', 'lobsun2', 'maghor2', 'malpar1', 'maltro1', 'malwoo1', 'marsan', 'mawthr1', 'moipig1', 'nilfly2', 'niwpig1', 'nutman', 'orihob2', 'oripip1', 'pabflo1', 'paisto1', 'piebus1', 'piekin1', 'placuc3', 'plaflo1', 'plapri1', 'plhpar1', 'pomgrp2', 'purher1', 'pursun3', 'pursun4', 'purswa3', 'putbab1', 'redspu1', 'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar', 'rossta2', 'rufbab3', 'ruftre2', 'rufwoo2', 'rutfly6', 'sbeowl1', 'scamin3', 'shikra1', 'smamin1', 'sohmyn1', 'spepic1', 'spodov', 'spoowl1', 'sqtbul1', 'stbkin1', 'sttwoo1', 'thbwar1', 'tibfly3', 'tilwar1', 'vefnut1', 'vehpar1', 'wbbfly1', 'wemhar1', 'whbbul2', 'whbsho3', 'whbtre1', 'whbwag1', 'whbwat1', 'whbwoo2', 'whcbar1', 'whiter2', 'whrmun', 'whtkin2', 'woosan', 'wynlau1', 'yebbab1', 'yebbul3', 'zitcis1']
species_to_index = {species[i]:i for i in range(len(species))}
data['index_label'] = data['primary_label'].apply(lambda x: species_to_index[x])
data['tensor_label'] = pd.Series(pd.get_dummies(data['primary_label']).astype(int).values.tolist()).apply(lambda x: torch.Tensor(x))

# Remove overly short clips
data = data[data['duration'] >= MIN_SAMPLE_LENGTH]

# Use 100 rows of data for quick runs to test functionalities
if ABRIDGED_RUN == True:
    data = data.sample(100)

# Progress bars for loading data to memory
tqdm.pandas()

# print("Loading nocall background noise snippets into memory")
# nocalls = pd.read_csv(f"{DATA_DIR}nocall_snippets/filenames.csv")
# def load_nocall(filepath):
#    signal, _ = torchaudio.load(filepath)
    # Reduce to one channel
#    signal = torch.mean(signal,dim = 0).unsqueeze(0)
#    return signal
# nocalls['signal'] = nocalls['filename'].progress_apply(lambda x: load_nocall(f"{DATA_DIR}nocall_snippets/{x}"))
# print("Done")

print("Loading training audio signals into memory")
# Loads signal to memory, pads to 5 seconds, cuts to 60 seconds
def filepath_to_signal(filepath):
    signal, _ = torchaudio.load(filepath)
    return signal
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

# Taken from https://www.earthinversion.com/datascience/pink_noise_vs_white_noise/
def generate_pink_noise(samples):
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]

    pink_noise = np.random.randn(samples)
    pink_noise = np.convolve(pink_noise, b)
    pink_noise = np.convolve(pink_noise, a, mode='valid')
    return torch.Tensor(pink_noise).unsqueeze(0)

# Signal is a torch tensor of shape [1, _]
# snr is signal to noise ratio in dB, a float
def add_pink_noise(signal, snr):
    pink = generate_pink_noise(signal.shape[1])
    return torchaudio.functional.add_noise(signal, pink, torch.Tensor([snr]))

# def add_bg_noise(signal, snr):
#    bg_noise = nocalls['signal'].sample(1).iloc[0]
#    return torchaudio.functional.add_noise(signal, bg_noise, torch.Tensor([snr]))

################################################################################
# IMPORT BIRDCALL DETECTION MODEL
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

# Set device we'll train on    
device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")

detection_model = BirdCallDetector().to(device)
detection_model.load_state_dict(torch.load(CHECKPOINT_DIR + DETECTION_MODEL + '/final.pt', map_location=torch.device(device)))
detection_model.eval()

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
def slices(seq, window_size = SAMPLE_RATE*SAMPLE_LENGTH, stride = None, align_left = True):
    # If one window is larger than the sequence, just return the scraps or nothing
    if window_size > seq.shape[0]:
        if seq.shape[0] < SAMPLE_RATE*MIN_SAMPLE_LENGTH:
            return []
        pad_length = SAMPLE_RATE*SAMPLE_LENGTH - seq.shape[0]
        if align_left == False:
            seq = torch.flip(seq, [0])
            seq = torch.nn.functional.pad(seq, (0, pad_length))
            return [torch.flip(seq, [0])]
        return [torch.nn.functional.pad(seq, (0, pad_length))]
        
    # If stride is None, it defaults to window_size
    if stride == None:
        stride = window_size

    index_slices = []
    left_pointer = 0
    while left_pointer + window_size <= seq.shape[0]:
        index_slices += [[left_pointer, left_pointer + window_size]]
        left_pointer += stride

    if align_left == False:
        offset = seq.shape[0]-(left_pointer-stride)-window_size
        index_slices = [[a+offset, b+offset] for [a,b] in index_slices]

    result = [seq[a:b] for [a,b] in index_slices]

    if align_left == True and left_pointer < seq.shape[0]:
        scrap = seq[left_pointer : seq.shape[0]]
        pad_length = window_size - (seq.shape[0]-left_pointer)
        scrap = torch.nn.functional.pad(scrap, (0, pad_length))
        result.append(scrap)

    elif align_left == False and index_slices[0][1] > stride:
        scrap = seq[0: index_slices[0][1]-stride]
        scrap = torch.flip(scrap, [0])
        pad_length = window_size - (index_slices[0][1]-stride)
        scrap = torch.nn.functional.pad(scrap, (0, pad_length))
        result = [torch.flip(scrap, [0])] + result
    
    return result

# Note: signals and labels should be ordinary lists
# Training boolean is used to decide whether to apply masks
# Config should have the format of a dictionary
class BirdDataset(Dataset):
    def __init__(self, signals, labels, 
                 training = True,
                 use_mel = True,
                 use_time_mask = True, 
                 use_freq_mask = True, 
                 use_pink_noise = True):
        super().__init__()
        self.training = training
        self.use_mel = use_mel
        self.use_time_mask = use_time_mask
        self.use_freq_mask = use_freq_mask
        self.use_pink_noise = use_pink_noise
        print(f'Preprocessing {"training" if training else "validation"} data\n')
        self.processed_clips, self.labels = self.process(signals, labels)

    def process(self, signals, labels):
        results = []
        new_labels = []
        for i, signal in enumerate(tqdm(signals, total = len(signals), leave = False)):
            # Cut signal into 5 second chunks and process each clip separately
            for clip in slices(signal.squeeze()):
                x = clip.unsqueeze(0)
                results += [x]
                x = spectrogram_transform(x)
                x = mel_spectrogram_transform(x)
                x = db_scaler(x)
                x = resize(x)
                x = x.to(device)
                prob_birdcall = detection_model(x.unsqueeze(0))[0][1].item()
                if prob_birdcall >= 0.5:
                    new_labels += [labels[i]]
                else:
                    new_labels += [torch.zeros(len(species))]
        return results, new_labels

    def __len__(self):
        return len(self.processed_clips)

    def __getitem__(self, index):
        # Process clip to tensor
        x = self.processed_clips[index]

        # Add pink noise
        if self.use_pink_noise and self.training:
            x = add_pink_noise(x, np.random.uniform(low = MIN_SNR, high = MAX_SNR))

        # Process
        x = spectrogram_transform(x)
        if self.use_mel:
            x = mel_spectrogram_transform(x)
        if self.training:
            exponent = np.random.uniform(low = 0.5, high = 3)
            x = torch.pow(x, exponent)
        x = db_scaler(x)
        if self.use_time_mask and self.training:
            x = time_mask(x)
        if self.use_freq_mask and self.training:
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
            nn.Dropout()
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
            nn.Dropout()
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

#label_counts = data_train['primary_label'].value_counts()

#total_durs = data_train.groupby(by='primary_label')['duration'].sum()
#total_durs = data_train['primary_label'].apply(lambda x: total_durs[x])

#weights = data_train['primary_label'].apply(lambda x: 1/label_counts.loc[x])*(data_train['duration']/total_durs)
#weights = data_train['duration']/total_durs
#weighted_sampler = torch.utils.data.WeightedRandomSampler(weights.to_list(), len(weights))

################################################################################
# TRAINING SETUP
################################################################################

# Create a saving directory if needed
output_dir = Path(f'{CHECKPOINT_DIR}{MODEL_NAME}')
output_dir.mkdir(parents=True, exist_ok=True)
output_dir = f'{CHECKPOINT_DIR}{MODEL_NAME}'

# Instantiate our training dataset
train_dataset = BirdDataset(signals = data_train['signal'].to_list(), 
                            labels = data_train['tensor_label'].to_list(),
                            training = True,
                            use_pink_noise = True)
train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

# Instantiate our validation dataset
validation_dataset =  BirdDataset(signals = data_validation['signal'].to_list(), 
                                  labels = data_validation['tensor_label'].to_list(),
                                  training = False)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate our model
model = BirdClassifier(NUM_SPECIES).to(device)

# Set our loss function and optimizer
pos_weight = torch.ones([len(species)]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
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