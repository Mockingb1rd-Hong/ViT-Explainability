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
from my_explain import interpret_image, interpret_image_weighted, interpret_image_mine, interpret_image_ours

from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class CocoAnnotations:
    def __init__(self, annotation_file):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def get_annotation(self, image_id):
        return [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]

    def get_categories(self):
        return self.annotations['categories']

    def get_images(self):
        return self.annotations['images']


clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

# load the COCO annotations
coco = CocoAnnotations('C:/Hong/annotations/instances_val2014.json')

# load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print('device: ' + device)
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Define the COCO categories
coco_categories_text = [category['name'] for category in coco.get_categories()]
print(coco_categories_text)
coco_categories_tokens = model.encode_text(clip.tokenize(coco_categories_text).to(device))
# Create a dictionary mapping from category_id to category_name
category_id_to_name = {category['id']: category['name'] for category in coco.get_categories()}

def perturb_image(image, image_relevance, percentage):
    # Flatten the relevance scores to a 1D array
    relevance_flat = image_relevance.flatten()

    # Calculate the number of tokens to remove based on the percentage
    num_tokens = relevance_flat.shape[0]
    num_to_remove = int(num_tokens * percentage)

    # Find the indices of the most important tokens based on relevance scores
    indices_to_remove = torch.topk(relevance_flat, num_to_remove, largest=True).indices

    # Create a binary mask for the tokens to remove
    binary_mask = torch.zeros_like(relevance_flat, dtype=torch.bool)
    binary_mask[indices_to_remove] = True

    # Reshape the binary mask to the original image relevance shape
    binary_mask = binary_mask.view_as(image_relevance)

    # Upsample the binary mask to match the image dimensions
    binary_mask = torch.nn.functional.interpolate(binary_mask.unsqueeze(0).unsqueeze(0).float(), size=(image.shape[2], image.shape[3]), mode='nearest').bool()

    # Expand the binary mask to match the image dimensions
    binary_mask = binary_mask.expand(image.shape[0], image.shape[1], -1, -1)

    # Apply the binary mask to the image by setting the selected tokens to zero
    perturbed_image = image.clone()
    perturbed_image[binary_mask] = 0

    return perturbed_image

# Function to classify an image
def classify_image(image, top_k=5):
    # get the image features
    image_features = model.encode_image(image)

    # compare the image features to the category features
    similarities = (100.0 * image_features @ coco_categories_tokens.T).softmax(dim=-1)
    _, indices = similarities[0].topk(top_k)

    # return top-k predictions
    return [coco_categories_text[idx] for idx in indices]


def get_ground_truth(image_info):
    # Get the annotations for the image
    annotations = coco.get_annotation(image_info['id'])

    # Get the category names for the annotations
    ground_truth = [category_id_to_name[ann['category_id']] for ann in annotations]

    return ground_truth

def forward_hook(module, input, output):
    module.input = input[0]
    module.output = output

def register_hooks(model):
    hooks = []
    for blk in model.visual.transformer.resblocks.children():
        hook = blk.register_forward_hook(forward_hook)
        hooks.append(hook)
    return hooks



# path to the COCO val2014 images
image_dir = 'C:/Hong/val2014/val2014'

# get the image ids
image_infos = coco.get_images()
no_object = 0


sample_size = 5000
if len(image_infos) > sample_size and sample_size != -1:
    selected_image_infos = random.sample(image_infos, sample_size)
else:
    selected_image_infos = image_infos

# selected_image_infos = image_infos

steps = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 1]

correct_predictions = 0
total_predictions = {step: 0 for step in steps}
h_p_correct_predictions = {step: 0 for step in steps}
w_p_correct_predictions = {step: 0 for step in steps}
o_p_correct_predictions = {step: 0 for step in steps}
# m_p_correct_predictions = {step: 0 for step in steps}

pbar = tqdm(total=len(selected_image_infos), desc="Classifying images", dynamic_ncols=True)

# loop over the image infos
for image_info in selected_image_infos:
    annotations = coco.get_annotation(image_info['id'])
    # construct the image path
    image_path = os.path.join(image_dir, image_info['file_name'])
    # print(image_path)

    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # preprocess the image
    image = preprocess(image).unsqueeze(0).to(device)

    # classify the image
    # print(f"Classifying image {image_info['file_name']}")
    predictions = classify_image(image)
    ground_truth = get_ground_truth(image_info)
    # print(ground_truth)
    if len(ground_truth) >= 1:
        texts = ground_truth[0]
    else:
        no_object += 1
        pbar.update(1)
        continue
    text = clip.tokenize(texts).to(device)


    hooks = register_hooks(model)
    R_image = interpret_image(model=model, image=image, texts=text, device=device, start_layer=0)
    # R_image_mine = interpret_image_mine(model=model, image=image, texts=text, device=device, start_layer=0)
    R_image_ours = interpret_image_ours(model=model, image=image, texts=text, device=device)
    for hook in hooks:
       hook.remove()
    hooks = register_hooks(model)
    R_image_weighted = interpret_image_weighted(model=model, image=image, texts=text, device=device, start_layer=0)
    for hook in hooks:
       hook.remove()

    for step in steps:
        h_perturbated_image = perturb_image(image, R_image, step)
        h_p_predictions = classify_image(h_perturbated_image)

        o_perturbated_image = perturb_image(image, R_image_ours, step)
        o_p_predictions = classify_image(o_perturbated_image)

        w_perturbated_image = perturb_image(image, R_image_weighted, step)
        w_p_predictions = classify_image(w_perturbated_image)

        # m_perturbated_image = perturb_image(image, R_image_mine, step)
        # m_p_predictions = classify_image(m_perturbated_image)

        # Increment correct predictions if the number of correct predictions equals or exceeds the size of ground truth
        if texts in h_p_predictions:
            h_p_correct_predictions[step] += 1
        if texts in w_p_predictions:
            w_p_correct_predictions[step] += 1
        if texts in o_p_predictions:
            o_p_correct_predictions[step] += 1
        # if texts in m_p_predictions:
        #     m_p_correct_predictions[step] += 1

        total_predictions[step] += 1


    # print()
    pbar.update(1)

pbar.close()

# print(w_p_predictions)
# print(w_p_correct_predictions)
# compute accuracy
h_p_accuracy = {step: h_p_correct_predictions[step] / total_predictions[step] for step in steps}
w_p_accuracy = {step: w_p_correct_predictions[step] / total_predictions[step] for step in steps}
o_p_accuracy = {step: o_p_correct_predictions[step] / total_predictions[step] for step in steps}
# m_p_accuracy = {step: m_p_correct_predictions[step] / total_predictions[step] for step in steps}

# Store accuracies in arrays
h_p_accuracy_array = [h_p_accuracy[step] for step in steps]
w_p_accuracy_array = [w_p_accuracy[step] for step in steps]
o_p_accuracy_array = [o_p_accuracy[step] for step in steps]
# m_p_accuracy_array = [m_p_accuracy[step] for step in steps]

print(f"No Objects: {no_object}")
print(f"Chefer Accuracy: {h_p_accuracy_array}")
print(f"Huang Accuracy: {w_p_accuracy_array}")
print(f"Ours Accuracy: {o_p_accuracy_array}")
# print(f"Mine Accuracy: {m_p_accuracy_array}")

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(steps, h_p_accuracy_array, label='Chefer Accuracy', marker='o')
plt.plot(steps, w_p_accuracy_array, label='Huang Accuracy', marker='o')
plt.plot(steps, o_p_accuracy_array, label='Ours Accuracy', marker='o')
# plt.plot(steps, m_p_accuracy_array, label='Mine Accuracy', marker='o')

plt.xlabel('Perturbation Steps')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Perturbation Steps')
plt.legend()
plt.grid(True)
plt.show()