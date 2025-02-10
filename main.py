# Import Libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from sklearn.utils.class_weight import compute_class_weight

# Checking for Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Transforms with Data Augmentation
transformer = transforms.Compose([
    transforms.Resize((150, 150)),  # Updated resize to 150x150
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1,1]
])

# Dataloader
train_path = '../New_Plant_Diseases_Dataset/train'
test_path = '../New_Plant_Diseases_Dataset/test'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=64, shuffle=True
)

# Categories
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(f"Classes: {classes}")
num_classes = 38  # Set number of classes

# Calculate Class Weights
train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transformer)
class_weights = compute_class_weight('balanced', classes=np.arange(len(classes)), y=[label for _, label in train_dataset])
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define Loss Function with Class Weights
loss_function = nn.CrossEntropyLoss(weight=weights)

# CNN Model
class ConvNet(nn.Module):
    def __init__(self, num_classes=38):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        
        # Update in_features based on resized image dimensions
        self.fc1 = nn.Linear(in_features=64 * 18 * 18, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Initialize Model
model = ConvNet(num_classes=num_classes).to(device)

# Optimizer and Learning Rate Scheduler
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training Configuration
num_epochs = 20
train_count = len(glob.glob(train_path + '/**/*.[jp][pn]g', recursive=True))
test_count = len(glob.glob(test_path + '/**/*.[jp][pn]g', recursive=True))

print(f"Train Images: {train_count}, Test Images: {test_count}")

# Training Loop
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_accuracy += int(torch.sum(prediction == labels.data))
    
    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count
    
    # Validation Loop
    model.eval()
    test_accuracy = 0.0
    
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))
    
    test_accuracy = test_accuracy / test_count
    scheduler.step()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Save Best Model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy
