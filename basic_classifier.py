import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# --------------------------------------------
# Define a Simple Vision-Language Model
# --------------------------------------------

class VisionLanguageModel(nn.Module):
    def __init__(self, num_classes=10, embed_dim=128):
        super(VisionLanguageModel, self).__init__()
        # Image encoder: a simple CNN
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 8, 8]
            nn.Flatten(),     # [B, 64*8*8]
            nn.Linear(64 * 8 * 8, embed_dim),
            nn.ReLU()
        )
        # Text encoder: an embedding layer mapping class indices to vectors
        self.text_encoder = nn.Embedding(num_classes, embed_dim)

    def forward(self, images, labels):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(labels)
        return image_embeds, text_embeds


# --------------------------------------------
# Contrastive Loss Function
# --------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    A simple contrastive loss (InfoNCE style) that uses a temperature parameter.
    It computes the cosine similarities between image and text embeddings,
    then applies cross-entropy loss.
    """
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_embeds, text_embeds):
        # Normalize embeddings
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        # Compute similarity matrix (batch_size x batch_size)
        logits = image_embeds @ text_embeds.t() / self.temperature
        # Ground-truth: matching pairs lie on the diagonal
        labels = torch.arange(image_embeds.size(0)).to(image_embeds.device)
        loss_i = self.criterion(logits, labels)
        loss_t = self.criterion(logits.t(), labels)
        return (loss_i + loss_t) / 2


def evaluate(model, dataloader, device):
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            image_embeds, text_embeds = model(images, labels)
            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)
            all_labels.append(labels)

    all_image_embeds = torch.cat(all_image_embeds)
    all_text_embeds = torch.cat(all_text_embeds)
    all_labels = torch.cat(all_labels)

    # Compute cosine similarity matrix between all image and text embeddings
    sim_matrix = all_image_embeds @ all_text_embeds.t()
    # For each image, predict the text with the highest similarity
    preds = sim_matrix.argmax(dim=1)
    # Compare the predicted indices with the ground truth indices
    correct = (all_labels == all_labels[preds]).sum().item()
    accuracy = correct / len(all_labels)
    print(f"Retrieval Accuracy: {accuracy*100:.2f}%")


def plot_tsne(embeds, labels, title, filename):
    tsne = TSNE(n_components=2)
    embeds_2d = tsne.fit_transform(embeds)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # --------------------------------------------
    # Device Setup and Data Preparation
    # --------------------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------------------------
    # Model, Loss, Optimizer
    # --------------------------------------------
    model = VisionLanguageModel(num_classes=10, embed_dim=128).to(device)
    criterion = ContrastiveLoss(temperature=0.05)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    NUM_EPOCHS = 10

    # --------------------------------------------
    # Training Loop
    # --------------------------------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            image_embeds, text_embeds = model(images, labels)
            loss = criterion(image_embeds, text_embeds)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

    # --------------------------------------------
    # Evaluation
    # --------------------------------------------
    evaluate(model, val_loader, device)

    # --------------------------------------------
    # TSNE Visualization of Embeddings
    # --------------------------------------------
    model.eval()
    all_image_embeds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            image_embeds, _ = model(images, labels.to(device))
            all_image_embeds.append(image_embeds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_image_embeds = np.concatenate(all_image_embeds)
    all_labels = np.concatenate(all_labels)

    plot_tsne(all_image_embeds, all_labels, "CIFAR-10 Image Embeddings (TSNE)", "tsne_cifar10.png")
    print("TSNE plot saved as 'tsne_cifar10.png'")
