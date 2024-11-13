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
from tqdm import tqdm
import random

from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

'''
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

'''
def preprocess_image(image_path, preprocess):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = preprocess(image).unsqueeze(0)
    return image


def vqa_task(model, image, question, answer_choices, device):
    # Tokenize the question and answer choices
    question_tokens = clip.tokenize([question]).to(device)
    answer_tokens = clip.tokenize(answer_choices).to(device)

    # Debugging: Print shapes of the tokens
    # print(f"Question tokens shape: {question_tokens.shape}")
    # print(f"Answer tokens shape: {answer_tokens.shape}")

    # Check if answer_tokens is empty
    if answer_tokens.shape[0] == 0:
        # print(f"Skipping question '{question}' due to empty answer tokens.")
        # print(f"Answer choices: {answer_choices}")
        return None

    # Encode the image and answer choices
    image_features = model.encode_image(image)
    answer_features = model.encode_text(answer_tokens)

    # Compute similarity between the image and each answer
    similarity_scores = (image_features @ answer_features.T).squeeze(0)

    # Find the answer with the highest similarity score
    best_answer_index = similarity_scores.argmax().item()
    best_answer = answer_choices[best_answer_index]

    return best_answer

def evaluate_single_vqa_task(model, image, question, answer_choices, ground_truth_labels, device):
    best_answer = vqa_task(model, image, question, answer_choices, device)
    if best_answer is not None:
        # Check if the best answer is in the ground truth labels
        is_correct = best_answer in ground_truth_labels
    else:
        is_correct = False

    return is_correct, best_answer

'''
# Load the VQA data from the JSON file
with open('data/vqa/valid.json', 'r') as f:
    vqa_data = json.load(f)

# Select a random sample of entries
sample_size = 1000
if len(vqa_data) > sample_size:
    vqa_data_sample = random.sample(vqa_data, sample_size)
else:
    vqa_data_sample = vqa_data

correct_count = 0
total_count = 0
no_answer_count = 0

# Initialize the progress bar
pbar = tqdm(total=len(vqa_data_sample), desc="Evaluating VQA", dynamic_ncols=True)

# Process each entry
for entry in vqa_data_sample:
    image_path = f"C:/Hong/val2014/val2014/{entry['img_id']}.jpg"
    question = entry['sent']
    answer_choices = list(entry['label'].keys())
    ground_truth_labels = entry['label']

    # Preprocess the image
    image = preprocess_image(image_path, preprocess).to(device)

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
'''