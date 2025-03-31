import pdb
from tqdm import tqdm
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import CLIPProcessor, CLIPModel

from plotter import evaluate

# Add Cuda if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------
# Globals
# --------------------------------------------
# Training Variables
BATCH_SIZE = 32
NUM_EPOCHS = 10
LOSS_TEMP = 0.05
LEARNING_RATE = 1e-3
# Model Variables
EMBED_DIM = 128


# --------------------------------------------
# Tokenizer
# --------------------------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)

# --------------------------------------------
# Data Pre-processing
# --------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])


# --------------------------------------------
# Datasets: Fashion
# --------------------------------------------
ds = load_dataset('ceyda/fashion-products-small')
print(ds)
entry = ds['train'][0]
print(entry)


# --------------------------------------------
# SubCategories
# --------------------------------------------
dataset = ds['train']
subcategories = list(set(example['subCategory'] for example in dataset))
print("Number of subcategories:", len(subcategories))
print("subcategories:", subcategories)

# --------------------------------------------
# Split Datasets
# --------------------------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# --------------------------------------------
# Custom Fashion Dataset
# --------------------------------------------
class FashionDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        subcategory = item['subCategory']
        label = subcategories.index(subcategory)
        return self.transform(image), label


# --------------------------------------------
# DataLoaders
# --------------------------------------------
# Create DataLoader for training and validation sets
train_loader = DataLoader(FashionDataset(train_dataset), batch_size=32, shuffle=True)
val_loader = DataLoader(FashionDataset(val_dataset), batch_size=32, shuffle=False)


# --------------------------------------------
# CLIP Fine-tune Model
# --------------------------------------------
class CLIPFineTuner(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.vision_model.config.hidden_size, num_classes)

    def forward(self, x):
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(x)
            pooled_output = vision_outputs.pooler_output  # shape: [batch_size, hidden_size]
        return self.classifier(pooled_output)


# --------------------------------------------
# Define Model
# --------------------------------------------
# Finetune Model
num_classes = len(subcategories)
model_ft = CLIPFineTuner(clip_model, num_classes).to(device)


# --------------------------------------------
# Loss and Optimization
# --------------------------------------------
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


# --------------------------------------------
# Training
# --------------------------------------------
for epoch in range(NUM_EPOCHS):
    model_ft.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: 0.0000")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

    print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    model_ft.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

# Save model
torch.save(model_ft.state_dict(), 'clip_finetuned.pth')


# --------------------------------------------
# Inference
# --------------------------------------------
evaluate(model_ft, val_loader, subcategories, device, save_name="cm_clipfinetune_val.png")
evaluate(model_ft, val_loader, subcategories, device, save_name="cm_clipfinetune_train.png")
