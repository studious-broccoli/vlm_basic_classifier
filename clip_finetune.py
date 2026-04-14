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

# --------------------------------------------
# Globals
# --------------------------------------------
# Training Variables
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
# Model Variables
EMBED_DIM = 128


# --------------------------------------------
# Custom Fashion Dataset
# --------------------------------------------
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FashionDataset(Dataset):
    def __init__(self, data, train: bool = False):
        self.data = data
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
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
# CLIP Fine-tune Model
# --------------------------------------------
class CLIPFineTuner(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.clip_model = clip_model
        # Classifier operates on the projected embedding space so that
        # visual_projection (the layer we fine-tune) sits between the frozen
        # backbone and the classifier head.
        self.classifier = nn.Linear(clip_model.config.projection_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(x)
            pooled_output = vision_outputs.pooler_output  # [B, hidden_size]
        # visual_projection is outside no_grad — gradients flow through it
        projected = self.clip_model.visual_projection(pooled_output)  # [B, projection_dim]
        return self.classifier(projected)


def wise_ft(model: CLIPFineTuner, original_weights: dict, alpha: float = 0.5) -> None:
    """Interpolate visual_projection between fine-tuned and pretrained weights.

    Args:
        model: CLIPFineTuner after training.
        original_weights: Saved state dict of visual_projection before training.
        alpha: Weight given to the fine-tuned model (0 = pretrained, 1 = fine-tuned).
    """
    with torch.no_grad():
        for name, param in model.clip_model.visual_projection.named_parameters():
            orig = original_weights[name].to(param.device)
            param.data = alpha * param.data + (1 - alpha) * orig


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --------------------------------------------
    # Pretrained CLIP
    # --------------------------------------------
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)

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
    # DataLoaders
    # --------------------------------------------
    train_loader = DataLoader(FashionDataset(train_dataset, train=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FashionDataset(val_dataset, train=False), batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------------------------
    # Define Finetune Model
    # --------------------------------------------
    num_classes = len(subcategories)

    # Save visual_projection weights before any training so WiSE-FT can
    # interpolate back toward the pretrained values after training.
    original_proj_weights = {
        name: param.data.clone()
        for name, param in clip_model.visual_projection.named_parameters()
    }

    model_ft = CLIPFineTuner(clip_model, num_classes).to(device)

    # --------------------------------------------
    # Loss and Optimization
    # --------------------------------------------
    # visual_projection is trained at a lower LR than the classifier head so
    # the pretrained CLIP features are disturbed as little as possible.
    optimizer = optim.Adam([
        {"params": model_ft.clip_model.visual_projection.parameters(), "lr": 1e-5},
        {"params": model_ft.classifier.parameters(), "lr": 1e-4},
    ])
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

    # --------------------------------------------
    # WiSE-FT: interpolate visual_projection with pretrained weights
    # --------------------------------------------
    # alpha=0.5 is the default from the paper; tune toward 1.0 if in-distribution
    # accuracy matters more, toward 0.0 to preserve zero-shot robustness.
    wise_ft(model_ft, original_proj_weights, alpha=0.5)
    print("Applied WiSE-FT (alpha=0.5)")

    # Save model
    torch.save(model_ft.state_dict(), 'clip_finetuned.pth')

    # --------------------------------------------
    # Inference
    # --------------------------------------------
    evaluate(model_ft, val_loader, subcategories, device, save_name="cm_clipfinetune_val.png")
    evaluate(model_ft, train_loader, subcategories, device, save_name="cm_clipfinetune_train.png")
