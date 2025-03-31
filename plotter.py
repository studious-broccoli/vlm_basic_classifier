import pdb
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import pandas as pd


# --------------------------------------------
# TSNE Embedding Plot
# --------------------------------------------
def plot_tsne(embeds, labels, title, filename):
    tsne = TSNE(n_components=2)
    embeds_2d = tsne.fit_transform(embeds)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    plt.savefig(filename)
    plt.close()


# --------------------------------------------
# Confusion Matrix with Spec, Sens
# --------------------------------------------
def basic_cm_snsp(y_true, y_pred, title, filename, class_names=None):
    # Get unique class names (sorted for consistency)
    if class_names is None:
        class_names = sorted(set(y_true).union(set(y_pred)))

    class_names = sorted(set(y_true).union(set(y_pred)))
    num_classes = len(class_names)

    fig = plt.figure(f"Confusion Matrix", figsize=(10, 10))
    np.seterr(invalid='ignore')

    cf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)

    total_preds = np.sum(cf_matrix)
    tn = total_preds - cf_matrix.sum(axis=1)
    tp = [cf_matrix[i, i] for i in range(0, num_classes)]
    fp = cf_matrix.sum(axis=0) - tp
    specificity = tn / (tn + fp)

    # compute recall (sensitivity) for each class
    # R = TP / (TP + FN)
    denom = cf_matrix.sum(axis=1)[:, None]
    recall = cf_matrix / denom if denom.any() > 0. else 1.

    # only include precision and recall labels along matrix diagnoal
    diag_indx = [v * (num_classes + 1) for v in range(0, num_classes)]

    # pdb.set_trace()

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    group_recall = ["" if (i not in diag_indx or np.isnan(v)) else "Se: {0:.1%}".format(v) for i, v in enumerate(recall.flatten())]
    group_specificity = ["" if (i not in diag_indx) else "Sp: {0:.1%}".format(specificity[i % num_classes]) for i in range(0, num_classes * num_classes)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_counts, group_recall, group_specificity)]
    labels = np.asarray(labels).reshape(num_classes, num_classes)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(class_names, rotation=45)
    ax.yaxis.set_ticklabels(class_names, rotation=45)

    plt.title(title)

    # Display the visualization of the Confusion Matrix.
    print(f"saving Confusion Matrix")
    plt.savefig(filename)
    plt.close('all')


# --------------------------------------------
# Plot Sample Images
# --------------------------------------------
def plot_sample(dataset, indices, transform, device, model_ft, subcategories, save_name):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, idx in enumerate(indices):
        image, label_idx = dataset[idx]
        true_label = subcategories[label_idx]

        # Transform already applied by dataset
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_ft(image_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = subcategories[predicted_idx.item()]

        # Plot (convert tensor to numpy for display)
        axes[i].imshow(image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # unnormalize
        axes[i].set_title(f'True: {true_label}\nPredicted: {predicted_label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


# --------------------------------------------
# Plot Sample Images and CM from Known Text List
# --------------------------------------------
def plot_known_texts_acc(vlm, device, data_loader, known_labels, tokenizer, caption, image_name, show_image=False):
    vlm.to(device)
    vlm.eval()

    ytrue_labels = []
    ypred_labels = []
    losses_cosine = []

    # Step 2: Generate text embeddings for known texts
    text_embeds_list = []
    for text in known_labels:
        tokenized_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=12)
        input_ids = tokenized_input["input_ids"]
        attention_masks = tokenized_input["attention_mask"]

        with torch.no_grad():
            try:
                input_ids = input_ids.to(device)
                attention_mask = attention_masks.to(device)
                text_embeds = vlm.text_encoder(input_ids, attention_mask)
                text_embeds = text_embeds.squeeze(0)
                text_embeds_list.append(text_embeds)
            except:
                print(f"Error with {text}")

    # Iterate over the batches of data
    for batch in data_loader:
        pixel_values = batch["pixel_values"].to(device)  # Shape: (batch_size, channels, height, width)
        true_labels = batch[caption]  # Ground truth labels

        # Step 1: Generate the image embeddings for the entire batch
        with torch.no_grad():
            image_embeds = vlm.image_encoder(pixel_values)  # Shape: (batch_size, embed_dim)

        for i in range(image_embeds.shape[0]):  # Iterate over each image in the batch
            img_embed = image_embeds[i]  # Shape: (embed_dim)
            cosine_similarities = []

            for text_embeds in text_embeds_list:
                # Step 3: Compute cosine similarity between image embedding and each text embedding
                cosine_similarity = torch.cosine_similarity(img_embed, text_embeds, dim=0)
                cosine_similarities.append(cosine_similarity.item())

            # Step 4: Find the best match by cosine similarity
            best_match_idx = torch.tensor(cosine_similarities).argmax().item()
            best_match_text = known_labels[best_match_idx]
            best_similarity_score = cosine_similarities[best_match_idx]

            # Store results
            ytrue_labels.append(true_labels[i])
            ypred_labels.append(best_match_text)
            losses_cosine.append(best_similarity_score)

    # Convert true and predicted labels to indices to match the known_texts
    ytrue = [known_labels.index(y) for y in ytrue_labels]
    ypred = [known_labels.index(y) for y in ypred_labels]

    # Calculate accuracy using sklearn's accuracy_score function
    accuracy = accuracy_score(ytrue, ypred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix if needed
    # basic_cm_snsp(ytrue_labels, ypred_labels, image_name, show_image=show_image)
    basic_cm_snsp(ytrue_labels, ypred_labels, "", image_name, class_names=None)

    return accuracy


# --------------------------------------------
# plot Performance in Bar Graph
# --------------------------------------------
def perf_report(y_true, y_pred, save_name):
    class_names = sorted(set(y_true).union(set(y_pred)))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Convert to DataFrame for easier visualization or CSV export
    df = pd.DataFrame(report).transpose()
    df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(20, 6))
    plt.title("Precision / Recall / F1 by Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


# --------------------------------------------
# Evaluate and Plot CM and Bar Graphs
# --------------------------------------------
def evaluate(model_ft, data_loader, subcategories, device, save_name="cm_clipfinetune.png"):
    model_ft.eval()
    correct, total = 0, 0
    ytrue_labels = []
    ypred_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            ytrue_labels.extend(labels.cpu().numpy())
            ypred_labels.extend(predicted.cpu().numpy())

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # Convert indices to class names
    ytrue_text = [subcategories[i] for i in ytrue_labels]
    ypred_text = [subcategories[i] for i in ypred_labels]

    # Plot confusion matrix
    basic_cm_snsp(ytrue_text, ypred_text,
                  "CLIP Finetune Confusion Matrix",
                  save_name,
                  class_names=subcategories)

    perf_report(ytrue_text, ypred_text, save_name.replace("cm_", "bar_"))


# --------------------------------------------
#
# --------------------------------------------


# --------------------------------------------
#
# --------------------------------------------


# --------------------------------------------
#
# --------------------------------------------