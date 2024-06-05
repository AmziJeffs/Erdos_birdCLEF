import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess audio data
class BirdSongDataset(Dataset):
    def __init__(self, file_paths, labels, sr=22050, n_mfcc=13):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.le = LabelEncoder()
        self.labels_encoded = self.le.fit_transform(self.labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels_encoded[idx]
        audio, _ = librosa.load(file_path, sr=self.sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        mfccs = np.expand_dims(mfccs, axis=0)
        return torch.tensor(mfccs, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Example model architecture
class BirdSongModel(nn.Module):
    def __init__(self, num_classes):
        super(BirdSongModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(64 * 10, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1, x.size(2) * x.size(3))
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Load data
file_paths = ['path/to/audio1.wav', 'path/to/audio2.wav', ...]
labels = ['species1', 'species2', ...]

# Split data
X_train, X_val, y_train, y_val = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = BirdSongDataset(X_train, y_train)
val_dataset = BirdSongDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, loss function, and optimizer
num_classes = len(set(labels))
model = BirdSongModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Accuracy: {100 * correct / total}%')

# Save the model
torch.save(model.state_dict(), 'bird_song_model.pth')

# Predict
def predict(file_path, model, le, sr=22050, n_mfcc=13):
    model.eval()
    audio, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(mfccs)
        _, predicted = torch.max(outputs.data, 1)
    return le.inverse_transform(predicted.numpy())[0]

# Load the model and make a prediction
model.load_state_dict(torch.load('bird_song_model.pth'))
le = LabelEncoder()
le.fit(labels)
predicted_species = predict('path/to/new_bird_song.wav', model, le)
print(f'Predicted bird species: {predicted_species}')
