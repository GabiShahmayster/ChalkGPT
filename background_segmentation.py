import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import deque


class PatchEmbedder(nn.Module):
    def __init__(self, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove the final FC layer

    def forward(self, x):
        return self.backbone(x).squeeze()


def extract_patches(image, patch_size):
    patches = []
    for i in range(0, image.size[0] - patch_size + 1, patch_size):
        for j in range(0, image.size[1] - patch_size + 1, patch_size):
            patch = image.crop((i, j, i + patch_size, j + patch_size))
            patches.append(patch)
    return patches


def get_neighbors(index, n_cols, n_rows):
    row, col = divmod(index, n_cols)
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < n_rows and 0 <= c < n_cols:
            neighbors.append(r * n_cols + c)
    return neighbors


def region_growing(embeddings, similarity_threshold, n_cols, n_rows):
    n_patches = len(embeddings)
    labels = np.full(n_patches, -1)
    current_label = 0

    for seed in range(n_patches):
        if labels[seed] != -1:
            continue

        queue = deque([seed])
        labels[seed] = current_label

        while queue:
            current = queue.popleft()
            neighbors = get_neighbors(current, n_cols, n_rows)

            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    similarity = cosine_similarity(embeddings[current].reshape(1, -1),
                                                   embeddings[neighbor].reshape(1, -1))[0][0]
                    if similarity > similarity_threshold:
                        labels[neighbor] = current_label
                        queue.append(neighbor)

        current_label += 1

    return labels


def segment_background(image_path, patch_size=32, similarity_threshold=0.8, batch_size: int = 32):
    # Load and preprocess the image
    image = Image.open(image_path)
    patches = extract_patches(image, patch_size)
    n_cols = (image.size[0] - patch_size + 1) // patch_size
    n_rows = (image.size[1] - patch_size + 1) // patch_size

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchEmbedder(patch_size).to(device)
    model.eval()

    # Transform for input patches
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract embeddings
    embeddings = []
    n_batches: int = len(patches) // batch_size
    with torch.no_grad():
        for n in range(n_batches):
            start_idx: int = n*batch_size
            end_idx: int = min((n+1)*batch_size, len(patches))
            batched_tensor = None
            for patch in patches[start_idx:end_idx]:
                input_tensor = transform(patch).unsqueeze(0).to(device)
                if batched_tensor is None:
                    batched_tensor = input_tensor
                else:
                    batched_tensor = torch.concatenate((batched_tensor, input_tensor))
            embedding = model(batched_tensor).cpu().numpy()
            embeddings.append(embedding)

    embeddings = np.vstack(embeddings)

    # Perform region growing
    labels = region_growing(embeddings, similarity_threshold, n_cols, n_rows)

    # Identify the background label (assume it's the most common label)
    background_label = np.argmax(np.bincount(labels))

    # Create segmentation mask
    segmentation_mask = np.zeros(image.size[::-1], dtype=np.uint8)
    patch_index = 0
    for i in range(0, image.size[0] - patch_size + 1, patch_size):
        for j in range(0, image.size[1] - patch_size + 1, patch_size):
            if labels[patch_index] == background_label:
                segmentation_mask[j:j + patch_size, i:i + patch_size] = 255
            patch_index += 1

    return segmentation_mask


# Example usage
image_path = "/home/gabi/GitHub/Experiments/segment-anything-2/downloaded_frames_tag/011.jpg"
segmentation_mask = segment_background(image_path, similarity_threshold=0.9, patch_size=4, batch_size=64)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(Image.open(image_path))
ax1.set_title("Original Image")
ax1.axis('off')
ax2.imshow(segmentation_mask, cmap='gray')
ax2.set_title("Background Segmentation")
ax2.axis('off')
plt.tight_layout()
plt.show(block=True)