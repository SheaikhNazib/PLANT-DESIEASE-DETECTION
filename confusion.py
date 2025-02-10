import torch
import numpy as np
import pathlib
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Transforms
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Dataset Paths
train_path = '../New_Plant_Diseases_Dataset/train'
test_path = '../New_Plant_Diseases_Dataset/test'

# Data Loaders
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=64,  # Changed batch size to 64
    shuffle=True
)

# Classes
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
num_classes = len(classes)
print(f"Classes: {classes}")

# Define Model (if not already defined)
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
        
        self.dropout = nn.Dropout(0.5)
        
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

# Load Pretrained Model with weights_only=True to prevent unpickling arbitrary code
try:
    model.load_state_dict(torch.load('best_checkpoint.model', weights_only=True))
    print("Loaded best model checkpoint.")
except FileNotFoundError:
    print("Checkpoint not found. Please train the model or use a valid checkpoint.")

# Evaluation: Calculate Precision, Recall, F1-Score
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        # Store labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Convert results to numpy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Calculate Metrics
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

# Print Results
print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)

# Rotate the x-axis labels (prediction labels) vertically
plt.xticks(rotation=90)  # Rotate x-axis labels to make them vertical

plt.title("Confusion Matrix")
plt.show()
