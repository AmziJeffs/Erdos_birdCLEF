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
BATCH_SIZE = 96 # Number of samples per batch while training our network
NUM_EPOCHS = 10 # Number of epochs to train our network
LEARNING_RATE = 0.001 # Learning rate for our optimizer

# Directories
CHECKPOINT_DIR = "checkpoints/" # Checkpoints, models, and training data will be saved here
DATA_DIR = "../data/"
AUDIO_DIR = DATA_DIR + "train_audio/"
UNLABELED_DIR = DATA_DIR + "unlabeled_soundscapes/"

# CURRENT PIPELINE:
#  - Split each audio file into 5 second clips
#  - Discard any scrap with duration less than 3.5s. Pad others.
#  - Run Birdcall detection on each clip and change labels appropriately
#  - Loss function is CrossEntropyLoss
#  - Train with freq/time masking, random power, and pink bg noise.
#  - Validate without freq/time masking, random power, or bg noise.
#  - Model output will be a vector of logits. Need to apply sigmoid to get probabilities.


MODEL_NAME = "CE_ALLDATA_PINKBG_WITHDETECT_RANDOMSAMPLE_PSEUDOLABEL"
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
import tensorflow as tf
import tensorflow_hub as hub
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

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(physical_devices[0], device_type='GPU')

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

################################################################################
# GENERATING PSEUDOLABELS ON UNLABELED DATA VIA TRANSFER LEARNING 
################################################################################

# Using ideas from "Transfer Learning with Pseudo Multi-Label Birdcall Classification for DS@GT BirdCLEF 2024" 

google_model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8')
google_labels_path = hub.resolve('https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/8') + "/assets/label.csv"

google_labels = pd.read_csv(google_labels_path)
labels_2021 = set(google_labels['ebird2021'])

mask = np.zeros(len(labels_2021))

for s in species:
    if s in labels_2021:
        index = google_labels.index[google_labels['ebird2021'] == s].tolist()[0]
        mask[index] = 1

unlabeled_soundscapes = sorted(os.listdir(UNLABELED_DIR))
step_size = SAMPLE_RATE * 5
count = 0 

if ABRIDGED_RUN == True:
    unlabeled_soundscapes = random.sample(unlabeled_soundscapes, 10)
    



for path in tqdm(unlabeled_soundscapes):
    waveform, sr = librosa.load(UNLABELED_DIR + path, sr=SAMPLE_RATE)
    waveform = waveform.astype(np.float32)
    dur = librosa.get_duration(y=waveform, sr=sr)
    
    
    if len(waveform) < 240 * SAMPLE_RATE:
        continue
    
    
    for i in range(4):
        start = np.random.randint(waveform.shape[0]-SAMPLE_RATE*SAMPLE_LENGTH+1)

        x = waveform[start:start + SAMPLE_RATE*SAMPLE_LENGTH]
        logits = google_model.infer_tf(x[np.newaxis, :])
        logits = logits['label'].numpy()[0]
        #logits = logits[mask]
        logits = np.multiply(logits, mask)
        pred_label = google_labels.iloc[[np.argmax(logits)]]['ebird2021'].item() 
        
        if pred_label in species: 
            print("Adding new row to metadata")
            count += 1
            new_row = {
            'filepath': UNLABELED_DIR + path,
            'primary_label': pred_label,
            'duration': dur
            }
            data = pd.concat([data, pd.DataFrame([new_row])])
    

print(f"Added {count} datapoints from unlabeled soundscape")

data['index_label'] = data['primary_label'].apply(lambda x: species_to_index[x])
data['tensor_label'] = pd.Series(pd.get_dummies(data['primary_label']).astype(int).values.tolist()).apply(lambda x: torch.Tensor(x))

# Remove overly short clips
data = data[data['duration'] >= MIN_SAMPLE_LENGTH]

data.to_csv('metadata_with_pseudolabels.csv')
