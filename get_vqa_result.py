import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization
import cv2
from tqdm import tqdm
from scipy.ndimage import binary_opening, binary_closing, label
import torch
import torchvision.transforms as transforms
from PIL import Image
import CLIP.clip as clip
import json
import os
import torch.nn.functional as F
import random
from vqa_task import evaluate_single_vqa_task
from my_explain import interpret_image, interpret_text, interpret

from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

# load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print('device: ' + device)
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


def perturb_image(image, image_relevance, top_k):
    # Ensure the image and relevance scores are on the same device
    device = image.device

    # Flatten the relevance scores to a 1D array
    relevance_flat = image_relevance.flatten()

    # Get the shape of the image (channels, height, width)
    channels, height, width = image.shape[1:]

    # Find the indices of the top-k most important tokens based on relevance scores
    indices_to_remove = torch.topk(relevance_flat, top_k, largest=True).indices

    # Create a binary mask for the tokens to remove
    binary_mask = torch.zeros_like(image, dtype=torch.bool)
    for index in indices_to_remove:
        unraveled_index = torch.unravel_index(index, (channels, height, width))
        binary_mask[0, unraveled_index[0], unraveled_index[1], unraveled_index[2]] = True

    # Apply the binary mask to the image by setting the selected tokens to zero
    perturbed_image = image.clone()
    perturbed_image[binary_mask] = 0

    return perturbed_image


with open('data/vqa/valid.json', 'r') as f:
    vqa_data = json.load(f)

# Select a random sample of entries
sample_size = 1000
if len(vqa_data) > sample_size:
    vqa_data_sample = random.sample(vqa_data, sample_size)
else:
    vqa_data_sample = vqa_data

steps = [0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 1]
correct_count = 0
total_count = 0
no_answer_count = 0
k = 50

# Initialize the progress bar
pbar = tqdm(total=len(vqa_data_sample), desc="Evaluating VQA", dynamic_ncols=True)

# Process each entry
for entry in vqa_data_sample:
    image_path = f"C:/Hong/val2014/val2014/{entry['img_id']}.jpg"
    question = entry['sent']
    answer_choices = list(entry['label'].keys())
    ground_truth_labels = entry['label']

    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # preprocess the image
    image = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    num_tokens = image_features.shape[1]
    # print(f"Number of tokens: {num_tokens}")
    question_tokens = clip.tokenize([question]).to(device)
    _, R_image = interpret(image, question_tokens, model, device)
    R_image = interpret_image(image, question_tokens, model, device)
    print(R_image.shape)
    # perturbated_image = perturb_image(image, R_image, k)

    is_correct, best_answer = evaluate_single_vqa_task(model, image, question, answer_choices, ground_truth_labels, device)
    if best_answer is not None:
        if is_correct:
            correct_count += 1
        total_count += 1
    else:
        no_answer_count += 1
        total_count += 1

    # Update the progress bar
    pbar.update(1)

# Close the progress bar
pbar.close()

# Calculate and print accuracy
if total_count > 0:
    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy:.2f}")
    print(f"No answer count: {no_answer_count}")
else:
    print("No valid questions processed.")
    accuracy = 0.0

print(f"Final Accuracy: {accuracy:.2f}")